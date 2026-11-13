# Gesture Mirror
My program predicts your gesture, currently only open and closed, and mirrors that onto a robotic hand.

## How it was made
This project was made using Onshape on the CAD side of things, there where many research papers read about maschine learning to apply that to my own model and the core software is Python and C++ for the ESP32 control.

## How to use this
1. Build the robotic hand
see: [assembling robotic hand](docs/assembly.md)

2. Upload the firmware to esp32
see: [uploading firmware](docs/firmware_upload.md)

3. Clone the source code to your device
```bash
git clone https://github.com/Toaster-lab/GestureMirror
```

4. Install the python dependencies
```bash
pip install -r software/requirements.txt
```

5. Start the main program 
```bash
python software/main.py
```
How to use main.py? See: [How to use the main.py](docs/main_guide.md)

Model limitations? See: [Model limitations](docs/model_limitations.md)

You can just experiment with this if you want without the robotic hand!

## Why did I make this?
* A passion for machine learning and for building things!
* First step into helping disabled people: Technologie should be there to help others! Making a robotic hand is the first step into biomedical engineering and using machine learning to automate that. That is why a hand, this is a way for me to also learn about how prosthetic are built.
* A learning experience: This is my first CAD project, my first project where I incooperate maschine learning into hardware(computer vision) and I learned how to write documentation!
* I saw so many people build these robotic / bionic hands and thought to myself, could I do that? So I tried! I thought something like this was impossible but through trying my best, most things are possible.
* I want to take this and work on it from time to time, not as intense as now, but I can always comeback and improve things. Changing how the hand is controlled, the design of the hand and maybe making a real bionic hand!


## AI DISCLOSURE
I used AI for research, learning faster, feedback and helping with file paths. For example instead of searching for the proper documentation on something I am trying to learn / fix, I ask Claude to help me understand the Error message and give me in one of my cases, the real documentation of PyTorch of this topic so I can fix this myself. Code or text was NOT written with AI, I only used it for checking my grammar in the documentation! AI was used once for the math side of things when designing the hand, I did not know how I can know what the ROM is on the finger, so I needed help! Using AI responsible and not just vibe coding was a big goal of this project and coding is way more fun!