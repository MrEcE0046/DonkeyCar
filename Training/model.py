import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    Ett skräddarsytt neuralt nätverk (CNN) för att extrahera funktioner ur kamerabilder.
    Detta är ögat som ser linjen och omvandlar pixlar till abstrakt förståelse.
    """
    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Kontrollera att indata är i formatet (Kanaler, Höjd, Bredd)
        assert len(observation_space.shape) == 3, "Förväntade observationer i (C, H, W)"

        n_input_channels = observation_space.shape[0]

        # --- CNN-ARKITEKTUR (Faltningslager) ---
        # Varje Conv2d-lager letar efter mönster. De tidiga lagren ser enkla linjer,
        # medan de senare lagren förstår komplexa former som skarpa svängar.
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Beräkna automatiskt hur många neuroner som behövs efter CNN-lagren
        with th.no_grad():
            sample_input = th.zeros((1, *observation_space.shape), dtype=th.float32)
            n_flatten = self.cnn(sample_input).shape[1]

        # --- LINJÄRT LAGER (Beslutsfattaren) ---
        # Tar den extraherade informationen och komprimerar den till 'features_dim' (512).
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            # Dropout hjälper till att förhindra överträning genom att 
            # slumpmässigt stänga av 10% av kopplingarna under träning.
            nn.Dropout(0.1),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Kör bilddata genom nätverket för att få ut egenskaper."""
        return self.linear(self.cnn(observations))