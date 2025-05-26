import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from model.gnn import load_gnn_model


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class GraphCheck(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        num_devices = torch.cuda.device_count()   
        max_memory = {}
        for i in range(num_devices):
            total_memory = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
            max_memory[i] = f"{max(total_memory - 2, 2)}GiB"     
        kwargs.update({
            "max_memory": max_memory,
            "device_map": "auto",
            "revision": "main",
        })
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )
        
        # Freezing LLM
        for name, param in model.named_parameters():
            param.requires_grad = False

        model.gradient_checkpointing_enable()

        self.model = model
    
        print('Finish loading LLM!!!')

        self.word_embedding = self.model.model.get_input_embeddings()

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)
        
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.word_embedding.weight.shape[1]),
        ).to(self.model.device)

        self.embed_dim = self.word_embedding.weight.shape[1]
        self.gnn_output = args.gnn_hidden_dim
              

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    # autocast
    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    # Graph Encoding
    def encode_graphs(self, data):
        claim_kg = data['claim_kg'].to(self.model.device)
        doc_kg = data['doc_kg'].to(self.model.device)
        
        claim_n_embeds, _ = self.graph_encoder(claim_kg.x, claim_kg.edge_index.long(), claim_kg.edge_attr)   
        doc_n_embeds, _ = self.graph_encoder(doc_kg.x, doc_kg.edge_index.long(), doc_kg.edge_attr)
  
        if claim_kg.batch is not None:  
            claim_embeds = scatter(claim_n_embeds, claim_kg.batch, dim=0, reduce='mean')  
        else:  
            claim_embeds = claim_n_embeds.mean(dim=0, keepdim=True)
            
        if doc_kg.batch is not None:  
            doc_embeds = scatter(doc_n_embeds, doc_kg.batch, dim=0, reduce='mean')  
        else:  
            doc_embeds = doc_n_embeds.mean(dim=0, keepdim=True)

        claim_embeds = scatter(claim_n_embeds, claim_kg.batch, dim=0, reduce='mean')
        doc_embeds = scatter(doc_n_embeds, doc_kg.batch, dim=0, reduce='mean')
        return claim_embeds, doc_embeds

    def forward(self, data):
        # prompt texts and corresponding labels
        texts = self.tokenizer(data["text"], add_special_tokens=False)
        labels = self.tokenizer(data["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].cuda())
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).cuda()).unsqueeze(0)

        # encode graphs of claims and graphs of documents separately
        claim_embeds, doc_embeds = self.encode_graphs(data)

        # projection
        claim_embeds = self.projector(claim_embeds) 
        doc_embeds = self.projector(doc_embeds)

        batch_size = len(data['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids   
            input_ids = texts.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            # print(f"claim_embeds shape: {claim_embeds.shape}")
            # print(f"doc_embeds shape: {doc_embeds.shape}")
            
            # if claim_embeds or doc_embeds is null
            if claim_embeds.size(0) == batch_size:
                claim_embedding = claim_embeds[i].unsqueeze(0)
            else:
                claim_embedding = torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)

            if doc_embeds.size(0) == batch_size:
                doc_embedding = doc_embeds[i].unsqueeze(0)
            else:
                doc_embedding = torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
            
            inputs_embeds = torch.cat([bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # padding
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, data):

        # encode prompt
        texts = self.tokenizer(data["text"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].cuda())
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).cuda()).unsqueeze(0)

        # encode graphs
        claim_embeds, doc_embeds = self.encode_graphs(data)
        # projection
        claim_embeds = self.projector(claim_embeds)
        doc_embeds = self.projector(doc_embeds)
        
        # data['id'] = [data['id']] if isinstance(data['id'], int) else data['id']
        batch_size = len(data['id'])

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = texts.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            # if claim_embeds or doc_embeds is null
            if claim_embeds.size(0) == batch_size:
                claim_embedding = claim_embeds[i].unsqueeze(0)
            else:
                claim_embedding = torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
            if doc_embeds.size(0) == batch_size:
                doc_embedding = doc_embeds[i].unsqueeze(0)
            else:
                doc_embedding = torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
            
            inputs_embeds = torch.cat([bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0)
            
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # padding
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True,
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(pred)
        return {'id': data['id'],
                'pred': pred,
                'label': data['label'],
                'text': data['text']}

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
