import modules.scripts as scripts
import gradio as gr
import os
import uuid 
import numpy as np

from modules.processing import process_images, Processed, fix_seed
from modules.shared import opts, cmd_opts, state
from modules import extra_networks

def build_prompts(character_prompt, lora_name, lora_strength, lora_step):
  prompts = []

  total_stg=lora_strength+.1
  for nn in np.arange(0,total_stg,lora_step):
    prompts.append(f"{character_prompt}, <lora:{lora_name}:{nn}> ,")  

  return prompts

def make_gif(frames, filename = None):
    if filename is None:
      filename = str(uuid.uuid4())

    outpath = "outputs/txt2img-images/txt2gif"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    first_frame, append_frames = frames[0], frames[1:]

    first_frame.save(f"{outpath}/{filename}.gif", format="GIF", append_images=append_frames,
               save_all=True, duration=250, loop=0)

    return first_frame

def main(p, lora_name, lora_strength, lora_step):
  character_prompt = p.prompt.strip().rstrip(',')
  frame_prompts = build_prompts(character_prompt, lora_name, lora_strength, lora_step)

  fix_seed(p)

  state.job_count = len(frame_prompts)

  imgs = []
  all_prompts = []
  infotexts = []

  for i in range(len(frame_prompts)):
    if state.interrupted:
      break

    p.prompt = frame_prompts[i]
    proc = process_images(p)

    if state.interrupted:
      break

    imgs += proc.images
    all_prompts += proc.all_prompts
    infotexts += proc.infotexts

  gif = [make_gif(imgs,lora_name)]

  imgs += gif

  return Processed(p, imgs, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)

class Script(scripts.Script):
    is_txt2img = False

    # Function to set title
    def title(self):
        return "LoRa-txt2gif"

    def ui(self, is_img2img):
        with gr.Row():
          lora_name = gr.Textbox(label="lora name", value = "")
        with gr.Row():
          lora_strength = gr.Number(label="lora Strength (final)", value = 1)
        with gr.Row():
          lora_step = gr.Number(label="lora Steps", value = .1)


        return [lora_name, lora_strength, lora_step]

    # Function to show the script
    def show(self, is_img2img):
        return True

    # Function to run the script
    def run(self, p, lora_name, lora_strength, lora_step):
        # Make a process_images Object        
        return main(p, lora_name, lora_strength, lora_step)
