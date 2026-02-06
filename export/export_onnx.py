import os
import sys
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from gymnasium import spaces
import onnx
import types

# --- MILJÖ & KOMPATIBILITET ---

# Robust Monkeypatch för Numpy 2.x -> 1.x kompatibilitet.
# Detta är kritiskt för Raspberry Pi OS (Bookworm) som ofta kör nyare Numpy-versioner
# än vad Stable Baselines 3 förväntar sig.
if not hasattr(np, '_core'):
    import numpy.core as core
    _core = types.ModuleType('numpy._core')
    _core.numeric = core.numeric
    _core.multiarray = core.multiarray
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core.numeric'] = core.numeric
    sys.modules['numpy._core.multiarray'] = core.multiarray
    print("[INFO] Numpy 2.x -> 1.x Monkeypatch applied.")

# Dynamisk sökvägshantering för att hitta projektmoduler (t.ex. 'training.model')
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# OBS: Säkerställ att mappen heter 'training' med små bokstäver för Linux-kompatibilitet
try:
    from training.model import CustomCNN
except ImportError:
    print("[ERROR] Kunde inte hitta 'training.model'. Kontrollera att mappen heter 'training' (små bokstäver).")
    sys.exit(1)

# --- ONNX EXPORT LOGIK ---

class DeterministicPPOPolicy(nn.Module):
    """
    Wrapper för att tvinga modellen att vara deterministisk vid körning (inferens).
    Istället för att sampla en rörelse väljs alltid medelvärdet (det mest sannolika).
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        
    def forward(self, obs: th.Tensor) -> th.Tensor:
        # PPO predikterar en distribution; vi extraherar medelvärdet för stabil körning.
        dist = self.policy.get_distribution(obs)
        return dist.distribution.mean

def export_to_onnx(model_path, onnx_path, features_dim=512):
    """Laddar SB3-checkpoint och konverterar till ONNX med optimerad arkitektur."""
    
    print(f"[PROCESS] Laddar SB3-modell från: {model_path}")
    
    # Laddar PPO-modellen med projektets specifika CNN-arkitektur
    # net_arch=[] innebär att vi kör en "flat" arkitektur efter CNN-lagret för lägre latens
    model = PPO.load(
        model_path,
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=features_dim),
            net_arch=[]
        ),
    )
    
    # Definierar observationsutrymmet: 12 kanaler (4 bilder stacking), 80px höjd, 160px bredd
    model.observation_space = spaces.Box(low=0, high=1, shape=(12, 80, 160), dtype=np.float32)
    
    # Flytta till CPU och sätt i utvärderingsläge (stänger av Dropout/BatchNorm)
    policy = model.policy.to("cpu")
    policy.eval()

    export_model = DeterministicPPOPolicy(policy)
    export_model.eval()
    
    # Dummy-input för att definiera nätverkets ingångsform för ONNX-kompilatorn
    dummy_input = th.randn(1, 12, 80, 160)
    
    # Rensa gamla exportfiler för att undvika korrupt data
    for f in [onnx_path, onnx_path + ".data"]:
        if os.path.exists(f):
            os.remove(f)
    
    temp_onnx = onnx_path + ".temp"
    
    print(f"[PHASE 1] Exporterar graf till {temp_onnx}...")
    th.onnx.export(
        export_model,
        (dummy_input,),
        temp_onnx, 
        opset_version=15, # Nyare opset för bättre stöd av komplexa operationer
        input_names=["input"],
        output_names=["action"],
        dynamic_axes={"input": {0: "batch_size"}, "action": {0: "batch_size"}},
    )
    
    print(f"[PHASE 2] Separerar vikter till extern .data-fil...")
    # Genom att spara som 'external data' håller vi själva .onnx-filen lättviktig
    loaded_model = onnx.load(temp_onnx)
    onnx.save_model(
        loaded_model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(onnx_path) + ".data",
        size_threshold=1024, # Tvingar splittring även för små lager
        convert_attribute=False
    )
    
    # Städa upp temporära filer
    for f in [temp_onnx, temp_onnx + ".data"]:
        if os.path.exists(f):
            os.remove(f)
    
    print(f"[SUCCESS] Modell exporterad: {onnx_path}")

if __name__ == "__main__":
    # Konsekventa filnamn med små bokstäver för att undvika fel på Raspberry Pi (Linux)
    best_file = os.path.join(project_root, "rl_model_1250000_steps.zip") # Ska träna vidare.... =)
    output_onnx = os.path.join(project_root, "line_follower.onnx")

    if os.path.exists(best_file):
        export_to_onnx(best_file, output_onnx, features_dim=512)
    else:
        print(f"[ERROR] Hittade inte filen: {best_file}")
        print("Kontrollera att filnamnet stämmer exakt (inkl. stora/små bokstäver).")