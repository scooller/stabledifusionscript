import modules.scripts as scripts
import gradio as gr
import os
import uuid 
import numpy as np

from modules.processing import process_images, Processed, fix_seed
from modules.shared import opts, cmd_opts, state
from modules import extra_networks

def build_prompts(character_prompt: str, lora_name: str, lora_strength: float, lora_step: float, extra_prompt: str):
  if lora_name is "" and extra_prompt is "":
    raise ValueError(f"LoRA Name and Extra Prompt Empty...")

  prompts: list[str] = []

  total_stg=lora_strength+.1  
  
  for nn in np.arange(0,total_stg,lora_step):
    negg=total_stg-nn;
    new_prompt = character_prompt.replace("-NGT", str(negg) )
    
    lora_prompt_suffix = "" if lora_name is "" else f", <lyco:{lora_name}:{nn}>, "
    extra_prompt_prefix = "" if extra_prompt is "" else f"({extra_prompt}:{nn}), "
    full_prompt = f"{extra_prompt_prefix}{new_prompt}{lora_prompt_suffix}"   
    prompts.append(full_prompt)
  return prompts

def make_gif(frames, filename = "", gif_time=250, gif_loop=False):
    if filename=="":
      filename = str(uuid.uuid4())

    outpath = "outputs/txt2img-images/txt2gif"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    outfile = f"{outpath}/{filename}.gif"
    counter = 0
    while os.path.exists(outfile) and counter < 1000:
      counter += 1
      outfile = f"{outpath}/{filename}_{counter:>04d}.gif"

    first_frame, append_frames = frames[0], frames[1:]
    
    gif_duration=gif_time/len(frames)
    
    g_loop = 0 if gif_loop else 1

    first_frame.save(outfile, format="GIF", append_images=append_frames,
               save_all=True, duration=gif_duration, loop=g_loop)
               
    print()
    print(f"Gif Created in {outpath}/{filename}.gif")
    print()

    return first_frame

def main(p, lora_name, lora_strength, lora_step, extra_prompt, gif_time, gif_loop):
  character_prompt = p.prompt.strip().rstrip(',')
  frame_prompts = build_prompts(character_prompt, lora_name, lora_strength, lora_step, extra_prompt)

  fix_seed(p)
  total_img=len(frame_prompts)

  state.job_count = total_img

  imgs = []
  all_prompts = []
  infotexts = []
  
  print()
  print(f'Creating gif with: {total_img} frames over {gif_time/1000} seconds')
  print()

  cNet=False
  
  for i in range(total_img):
    if state.interrupted:
      break

    p.prompt = frame_prompts[i]
    proc = process_images(p)

    if state.interrupted:
      break  
    
    if(len(proc.images)>1):
        imgs.append(proc.images[0])
        cNet=True
    else:
        imgs += proc.images
    all_prompts.append(proc.all_prompts)
    infotexts.append(proc.infotexts)

  gif = [make_gif(imgs,lora_name,gif_time,gif_loop)]
  if not cNet:
    imgs += gif
  
  return Processed(p, gif, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)

class Script(scripts.Script):
    is_txt2img = False

    # Function to set title
    def title(self):
        return "LoRA-txt2gif"

    def ui(self, is_img2img):
        with gr.Row():
          lora_name = gr.Textbox(label="LoRA name, (is put after the prompt)", value = "")
          extra_prompt = gr.Textbox(label="Extra Prompt (is put before the prompt)", value = "")
        with gr.Row():          
          lora_step = gr.Number(label="LoRA Steps", value = .1)
          lora_strength = gr.Number(label="LoRA Strength (final)", value = 1)
        with gr.Row():
          gif_time = gr.Number(label="Gif Duration in milliseconds (ms)", value = 3000)
          gif_loop = gr.Checkbox(label="Gif Loop", value = True, info="infinite reproduction?")
          # save_separate = gr.Checkbox(label="Save Each", value = True, info="Save each image?")
          # mp4_mode = gr.Checkbox(label="mp4", value = True, info="MP4?")
        with gr.Row():
          gr.HTML(label="Attention", value='<div class="svelte-vt1mxs gap panel" style="margin-top:10px">You Can use -NGT for reduce strength like -NGT is replace :1 to :0<br>example:<br>put in you prompt <b>(Red eyes:-NGT)</b></div>')

        return [lora_name, lora_strength, lora_step, extra_prompt, gif_time, gif_loop]

    # Function to show the script
    def show(self, is_img2img):
        return True

    # Function to run the script
    def run(self, p, lora_name, lora_strength, lora_step, extra_prompt, gif_time, gif_loop):
        # Make a process_images Object        
        return main(p, lora_name, lora_strength, lora_step, extra_prompt, gif_time, gif_loop)
