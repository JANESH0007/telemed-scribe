# 🚀 Training Person B's Model on Google Colab (Free GPU)

Because training on 1,000 examples takes 2 hours on a laptop CPU, we will use a **Free T4 GPU** on Google Colab to finish it in 3 minutes!

### Step 1: Zip your project folder
1. Zip this entire `telemed-scribe` folder.
2. Go to [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
3. In the top menu, click **Runtime > Change runtime type**, select **T4 GPU**, and click Save.

### Step 2: Upload and Unzip
1. On the left sidebar of Colab, click the **Folder icon** (Files).
2. Click the **Upload** icon and upload your `telemed-scribe.zip`.
3. In your first Notebook cell, run this command to unzip it:
```bash
!unzip telemed-scribe.zip -d telemed-scribe
```

### Step 3: Install Requirements
Create a new cell and run:
```bash
%cd telemed-scribe
!pip install -r requirements.txt
```

### Step 4: Run the Training Pipeline!
Create a new cell and run:
```bash
!python ml/extraction/train_lora.py
```
*(If you are inside the `ml/extraction` directory, you can also run `!python train_lora.py` directly).*

### Step 5: Download your Weights
When the training finishes (it will take ~3 minutes), a folder called `lora-medical-t5` will be created. 
1. In the left sidebar, right-click the `lora-medical-t5` folder and click **Download**.
2. Put that folder back into your local `telemed-scribe` directory on your laptop.
3. Run `python demo_person_b.py` locally and watch it extract medical entities perfectly!
