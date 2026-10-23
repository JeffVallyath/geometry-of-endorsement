"""The only model-specific semantic-free renderer and label adapter."""
import hashlib, json
MODEL_ID='google/gemma-2-27b-it'
EXPECTED={'model_type':'gemma2','num_hidden_layers':46,'hidden_size':4608,'intermediate_size':36864,
          'num_attention_heads':32,'num_key_value_heads':16,'head_dim':128,
          'attn_logit_softcapping':50.0,'final_logit_softcapping':30.0,
          'query_pre_attn_scalar':144,'sliding_window':4096,'max_position_embeddings':8192}

def gemma_messages(messages):
    """Preserve instruction and request bytes, separated by exactly two LFs."""
    if len(messages)!=2 or [x['role'] for x in messages]!=['system','user']:
        raise ValueError('EXPECTED_EXACTLY_SYSTEM_PLUS_USER')
    if any(not isinstance(x['content'],str) or not x['content'] for x in messages):
        raise ValueError('EMPTY_OR_NON_STRING_CONTENT')
    return [{'role':'user','content':messages[0]['content']+'\n\n'+messages[1]['content']}]

def encode_prompt(tok,messages,tokenize=True):
    return tok.apply_chat_template(gemma_messages(messages),tokenize=tokenize,add_generation_prompt=True)

def label_ids(tok):
    ids={word:tok.encode(word,add_special_tokens=False) for word in ('Yes','No')}
    if any(len(v)!=1 or tok.decode(v)!=k for k,v in ids.items()) or ids['Yes']==ids['No']:
        raise ValueError('YES_NO_MUST_BE_DISTINCT_SINGLE_TOKENS')
    end=tok.convert_tokens_to_ids('<end_of_turn>')
    if not isinstance(end,int) or end==tok.unk_token_id or tok.decode([end],skip_special_tokens=False)!='<end_of_turn>':
        raise ValueError('GEMMA_TURN_END_TOKEN_MISSING')
    return ids['Yes'][0],ids['No'][0],end

def encode_train(tok,row):
    messages=gemma_messages(row['request']['messages']);target=row['target']
    if target not in ('Yes','No'):raise ValueError('UNEXPECTED_TRAIN_LABEL')
    y,n,end=label_ids(tok)
    prefix=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    full=tok.apply_chat_template(messages+[{'role':'assistant','content':target}],tokenize=True)
    if full[:len(prefix)]!=prefix:raise ValueError('TRAIN_INFER_TEMPLATE_BOUNDARY')
    suffix=full[len(prefix):]
    if end not in suffix:raise ValueError('MISSING_TURN_END')
    # Official templates may put a newline after the turn separator; do not train it.
    suffix=suffix[:suffix.index(end)+1]
    if suffix!=[y if target=='Yes' else n,end]:raise ValueError('LABEL_CONTEXT_OR_SUFFIX_CHANGED')
    ids=prefix+suffix
    if len(ids)>3072:raise ValueError('NO_TRAINING_TRUNCATION')
    return {'input_ids':ids,'labels':[-100]*len(prefix)+suffix}

def validate_config(config):
    for key,value in EXPECTED.items():
        if config.get(key)!=value:raise ValueError(f'WRONG_GEMMA_ARCHITECTURE_OR_SEMANTICS:{key}')
    if config.get('tie_word_embeddings',True) is not True:raise ValueError('EXPECTED_TIED_EMBEDDINGS')
    return True

def assert_runtime_model(model):
    validate_config(model.config.to_dict())
    if model.config._attn_implementation!='eager':raise ValueError('EAGER_REQUIRED_TO_PRESERVE_SOFTCAPPING')
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    if base.get_input_embeddings().weight.data_ptr()!=base.get_output_embeddings().weight.data_ptr():
        raise ValueError('TIED_EMBEDDINGS_CHANGED')
    layers=base.model.layers
    if len(layers)!=46:raise ValueError('WRONG_LAYER_COUNT')
    for layer in layers:
        if layer.self_attn.attn_logit_softcapping!=50.0:raise ValueError('ATTENTION_SOFTCAP_DISABLED')
        if abs(layer.self_attn.scaling-(144**-0.5))>1e-12:raise ValueError('ATTENTION_SCALE_CHANGED')
    return True
