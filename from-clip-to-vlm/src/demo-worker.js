// Adapted from the official Hugging Face SmolVLM WebGPU example (Apache-2.0).
// One selected model per worker; switching terminates the previous worker.
let modelConfigs;
const configURL=new URL('./lab-models.json',import.meta.url);configURL.search=new URL(import.meta.url).search;
async function configurations(){return modelConfigs??=await fetch(configURL).then(r=>{if(!r.ok)throw Error('Model list could not be loaded.');return r.json();});}
const LIBRARY='https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
let api,processor,model,stopping,busy=false,interrupted=false;
const send=(type,data={})=>self.postMessage({type,...data});
self.addEventListener('message',async({data})=>{
 if(data.type==='stop'){interrupted=true;stopping?.interrupt();return;}
 if(busy)return;
 busy=true;
 try{
  if(data.type==='load'){
   const config=(await configurations()).find(m=>m.key===data.modelKey);
   if(!config)throw new Error('Choose a supported model.');
   const MODEL=config.modelId,REVISION=config.revision;
   api=await import(LIBRARY);
   stopping=new api.InterruptableStoppingCriteria();
   const opts={revision:REVISION,progress_callback:p=>send('progress',{progress:p})};
   processor=await api.AutoProcessor.from_pretrained(MODEL,opts);
   model=await api.AutoModelForVision2Seq.from_pretrained(MODEL,{...opts,device:'webgpu',dtype:{embed_tokens:'fp32',vision_encoder:'fp32',decoder_model_merged:'q4'}});
   send('ready',{modelKey:config.key,model:MODEL,revision:REVISION,library:'3.8.1',dtype:{embed_tokens:'fp32',vision_encoder:'fp32',decoder_model_merged:'q4'}});
  }
  if(data.type==='generate'){
   if(!model)throw new Error('Load the model first.');
   stopping.reset();interrupted=false;
   const begin=performance.now();
   send('stage',{message:'Read image pixels and prepare processor inputs.'});
   const image=await api.RawImage.read(data.image);
   const messages=[{role:'user',content:[{type:'image'},{type:'text',text:data.prompt}]}];
   const text=processor.apply_chat_template(messages,{add_generation_prompt:true});
   const inputs=await processor(text,[image],{do_image_splitting:false});
   const prepared=performance.now();
   send('stage',{message:'First forward pass: image encoding → visual context preparation → prompt prefill (grouped by the library, not individually timed).'});
   let first=null,raw='';
   const streamer=new api.TextStreamer(processor.tokenizer,{skip_prompt:true,skip_special_tokens:true,callback_function:chunk=>{first??=performance.now();raw+=chunk;send('chunk',{chunk});}});
   const output=await model.generate({...inputs,do_sample:false,max_new_tokens:64,repetition_penalty:1.1,streamer,stopping_criteria:stopping});
   const generated=output.slice(null,[inputs.input_ids.dims[1],null]);
   const final=processor.batch_decode(generated,{skip_special_tokens:true})[0];
   const end=performance.now();
   const generatedTokenIds=Array.from(generated.data,Number);
   const eos=model.generation_config?.eos_token_id??model.config?.eos_token_id??model.config?.text_config?.eos_token_id;
   const eosIds=Array.isArray(eos)?eos:[eos];
   const stopReason=interrupted?'user stop':eosIds.includes(generatedTokenIds.at(-1))?'EOS':generated.dims[1]>=64?'generation limit':'generation ended (EOS not exposed by config)';
   send('complete',{generatedTokenIds,stopReason,text:final,streamed:raw,ms:Math.round(end-begin),prepareMs:Math.round(prepared-begin),firstTextMs:first?Math.round(first-begin):null,outputTokens:generated.dims[1],generationMs:Math.round(end-prepared),settings:{do_sample:false,max_new_tokens:64,repetition_penalty:1.1,do_image_splitting:false}});
  }
 }catch(error){send('error',{message:error?.message||String(error)});}
 finally{busy=false;}
});
