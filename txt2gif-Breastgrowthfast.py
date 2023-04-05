import modules.scripts as scripts
import gradio as gr
import os
import uuid 

from modules.processing import process_images, Processed, fix_seed
from modules.shared import opts, cmd_opts, state

def build_prompts(character_prompt):
  prompts = []

  prompts.append(f"topless, (flat chest:1.2), nipples, {character_prompt}, ")
  prompts.append(f"topless, (flat chest:1), nipples, {character_prompt}, ")
  prompts.append(f"topless, (flat chest:.8), nipples, {character_prompt}, ")
  prompts.append(f"topless, (small breasts:1.2), nipples, {character_prompt}, ")
  prompts.append(f"topless, (small breasts:1), nipples, {character_prompt}, ")
  prompts.append(f"topless, (small breasts:.8), nipples, {character_prompt}, ")
  prompts.append(f"topless, (medium breasts:.8), nipples, {character_prompt}, ")
  prompts.append(f"topless, (medium breasts:1), nipples, {character_prompt}, ")
  prompts.append(f"topless, (medium breasts:1.2), nipples, {character_prompt}, ")
  prompts.append(f"topless, (large breasts:.8), nipples, {character_prompt}, ")
  prompts.append(f"topless, (large breasts:1), nipples, {character_prompt}, ")
  prompts.append(f"topless, (large breasts:1.2), nipples, hanging breasts, {character_prompt}, ")
  prompts.append(f"topless, (huge breasts:1), nipples, hanging breasts, {character_prompt}, ")
  prompts.append(f"topless, (huge breasts:1.2), nipples, hanging breasts, {character_prompt}, ")
  prompts.append(f"topless, (gigantic breasts:1), nipples, hanging breasts, {character_prompt}, ")
  prompts.append(f"topless, (gigantic breasts:1.2), nipples, hanging breasts, {character_prompt}, ")
  prompts.append(f"topless, (gigantic breasts:1.4), nipples, hanging breasts, {character_prompt}, ")
  prompts.append(f"topless, (gigantic breasts:1.6), nipples, hanging breasts, {character_prompt}, ")  
  prompts.append(f"topless, (gigantic breasts:1.4), nipples, hanging breasts, {character_prompt}, ")
  prompts.append(f"topless, (gigantic breasts:1.2), nipples, hanging breasts, {character_prompt}, ")
  prompts.append(f"topless, (gigantic breasts:1), nipples, hanging breasts, {character_prompt}, ")
  prompts.append(f"topless, (huge breasts:1.2), nipples, hanging breasts, {character_prompt}, ")
  prompts.append(f"topless, (huge breasts:1), nipples, hanging breasts, {character_prompt}, ")
  prompts.append(f"topless, (large breasts:1.2), nipples, hanging breasts, {character_prompt}, ")
  prompts.append(f"topless, (large breasts:1), nipples, {character_prompt}, ")
  prompts.append(f"topless, (large breasts:.8), nipples, {character_prompt}, ")
  prompts.append(f"topless, (medium breasts:1.2), nipples, {character_prompt}, ")
  prompts.append(f"topless, (medium breasts:1), nipples, {character_prompt}, ")
  prompts.append(f"topless, (medium breasts:.8), nipples, {character_prompt}, ")
  prompts.append(f"topless, (small breasts:.8), nipples, {character_prompt}, ")
  prompts.append(f"topless, (small breasts:1), nipples, {character_prompt}, ")
  prompts.append(f"topless, (small breasts:1.2), nipples, {character_prompt}, ")
  prompts.append(f"topless, (flat chest:.8), nipples, {character_prompt}, ")
  prompts.append(f"topless, (flat chest:1), nipples, {character_prompt}, ")
  prompts.append(f"topless, (flat chest:1.2), nipples, {character_prompt}, ")
  

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

def main(p):
  character_prompt = p.prompt.strip().rstrip(',')
  frame_prompts = build_prompts(character_prompt)

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

  gif = [make_gif(imgs)]

  imgs += gif

  return Processed(p, imgs, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)

class Script(scripts.Script):
    is_txt2img = False

    # Function to set title
    def title(self):
        return "txt2gif-Breast Growth Fast"


    # Function to show the script
    def show(self, is_img2img):
        return True

    # Function to run the script
    def run(self, p):
        # Make a process_images Object
        return main(p)
