import time

import torch
import torch.nn as nn

from quant import *
from sparsegpt import *
from modelutils import *
from datasets import load_from_disk

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False 


def get_inputs_and_labels(model, tokenizer, split='test', batch_size=1):
    device = model.device
    max_length = model.config.max_position_embeddings
    # test = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    test = load_from_disk("/home/jinyuanshi/workspace/obc/wikitext_data/")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    stride = 2048
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    counter = 0
    
    batch_input = []
    batch_target = []
    for begin_loc in range(0, seq_len, stride):
        counter += 1
        
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        batch_input.append(input_ids)
        batch_target.append(target_ids)
        
        if len(batch_input) < batch_size:
            continue
        else:
            inputs = torch.stack(batch_input, dim=0).squeeze(1)
            targets = torch.stack(batch_target, dim=0).squeeze(1)
            return inputs, targets


def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    losses = []
    cos_sims = []
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    # copy inps
    orig_inps = inps.clone()

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    orig_outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        # last layer continue
        if i >= len(layers) - 1:
            for j in range(args.nsamples):
                orig_outs[j] = layer(orig_inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            inps, outs = outs, inps
            orig_inps, orig_outs = orig_outs, orig_inps
            continue

        # find all nn.Linear in this layer
        subset = find_layers(layer)
        
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
              continue
            gpts[name] = SparseGPT(subset[name], args.sparsity, args.prunen, args.prunem, args.percdamp, args.blocksize, name=name)
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=True, mse=False
                )

        def record_teacher_batch(name):
            def tmp(_, inp, out):
                gpts[name].record_batch(inp[0].data, out.data)
            return tmp

        def calculate_h(name):
            def tmp(_, inp, out):
                gpts[name].calculate_h(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(record_teacher_batch(name)))

        # calculate teacher's output for this layer
        print('Calculating teacher output for layer %d ...' % i) 
        for j in range(args.nsamples):
            orig_outs[j] = layer(orig_inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        torch.cuda.empty_cache()

        # record teacher's output for weight reconstruction
        for name in gpts:
            gpts[name].teacher_output_datas = gpts[name].output_datas.cpu()
            gpts[name].free()
            torch.cuda.empty_cache()

        # remove original hooks
        for h in handles:
            h.remove()

        # register new hooks and reconstruct weights and prune
        names = []
        for name in gpts:
            names.append(name)
        
        torch.cuda.empty_cache()
        

        for idx, name in enumerate(names):
            handles[idx] = subset[name].register_forward_hook(calculate_h(name))
            print('Calculating student input for %s ...' % name)
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            print('Reconstruct weight for %s ...' % name)
            gpts[name].reconstruct_weight(gpts[name].input_datas, gpts[name].teacher_output_datas, ridge_regular=1e-10)
            # torch.cuda.empty_cache()
            # print('Fastprune for %s ...' % name)
            # gpts[name].fasterprune(
            #     args.sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            # )
            handles[idx].remove()
            gpts[name].free()
            gpts[name].teacher_output_datas = None
            del gpts[name]
            torch.cuda.empty_cache()
        
        del gpts

        torch.cuda.empty_cache()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        # calculate mse loss
        loss = torch.nn.functional.mse_loss(outs, orig_outs)
        losses.append(loss.item())
        del loss

        # calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(outs.view(args.nsamples, -1).float(), orig_outs.view(args.nsamples, -1).float(), dim=1)
        cos_sims.append(cos_sim.mean().item())
        del cos_sim


        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        orig_inps, orig_outs = orig_outs, orig_inps
    

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    nlls = []
    all_shift_logits = []
    orig_all_shift_logits = []
    for i in range(args.nsamples):
        hidden_states = inps[i].unsqueeze(0)
        orig_hidden_states = orig_inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
            orig_hidden_states = model.model.decoder.final_layer_norm(orig_hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
            orig_hidden_states = model.model.decoder.project_out(orig_hidden_states)
        hidden_states = hidden_states.cpu().float()
        orig_hidden_states = orig_hidden_states.cpu().float()
        model.lm_head = model.lm_head.cpu().float()
        lm_logits = model.lm_head(hidden_states)
        orig_lm_logits = model.lm_head(orig_hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        orig_shift_logits = orig_lm_logits[:, :-1, :].contiguous()
        all_shift_logits.append(shift_logits)
        orig_all_shift_logits.append(orig_shift_logits)
        shift_labels = dataloader[i][0][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (args.nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    model.lm_head = model.lm_head.to(dev).half()
    all_shift_logits = torch.cat(all_shift_logits, dim=0)
    orig_all_shift_logits = torch.cat(orig_all_shift_logits, dim=0)
    loss = torch.nn.functional.mse_loss(all_shift_logits, orig_all_shift_logits)
    losses.append(loss.item())
    del loss
    cos_sim = torch.nn.functional.cosine_similarity(
        all_shift_logits.view(args.nsamples, -1).float().cpu(), 
        orig_all_shift_logits.view(args.nsamples, -1).float().cpu(), 
        dim=1)
    cos_sims.append(cos_sim.mean().item())
    del cos_sim

    model.config.use_cache = use_cache
    print("Losses:")
    for loss in losses:
        print(loss)
    print("Cosine similarities:")
    for cos_sim in cos_sims:
        print(cos_sim)
    print("=====")

@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    # testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        # print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
         wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


def evaluation_inputs_and_labels(inputs, targets, model):       
    model = model.to(DEV)
    with torch.no_grad():
        inputs = inputs.to(model.device)
        targets = targets.to(model.device)
        outputs = model(inputs, labels=targets, output_attentions=False)
        # attention = outputs.attentions

    ppl = torch.exp(outputs.loss.sum())
    print(f'perplexity: {ppl}')
    return None


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str, 
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
       '--invert', action='store_true', 
       help='Invert subset.'
    )
    parser.add_argument(
       '--save', type=str, default='',
       help='Path to saved model.'
    )
    parser.add_argument(
       '--log_wandb', action='store_true',
       help='Whether to log to wandb.'
    )

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_opt(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    # input_ids, targets = get_inputs_and_labels(model=model, tokenizer=tokenizer, split='train', batch_size=args.nsamples)
    # dataloader = []
    # for i in range(args.nsamples):
    #     dataloader.append((input_ids[i].unsqueeze(0), targets[i].unsqueeze(0)))
    
    # model.eval()
    # print("Evaluating on train data...")
    # evaluation_inputs_and_labels(input_ids, targets, model)

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        print(time.time() - tick)

    # print("Evaluating on train data...")
    # model.eval()
    # evaluation_inputs_and_labels(input_ids, targets, model)

    train_input_ids = []
    for batch in dataloader:
        train_input_ids.append(batch[0])
    train_input_ids = torch.cat(train_input_ids, dim=0).reshape(1, -1)
    opt_eval(model, train_input_ids, DEV, args.dataset, args.log_wandb)
    test_input_ids = testloader.input_ids
    opt_eval(model, test_input_ids, DEV, args.dataset, args.log_wandb)
    # opt_eval(model, testloader, DEV, args.dataset, args.log_wandb)
    # for dataset in ['wikitext2']:
    #     dataloader, testloader = get_loaders(
    #         dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
    #     )
    #     print(dataset)
    #     opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)
