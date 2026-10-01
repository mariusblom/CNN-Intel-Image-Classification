graph TD
    subgraph Data_Pipeline [1. Daten-Pipeline]
        A[Intel Image Dataset<br/>150x150 px | 6 Klassen] --> B[Data Preprocessing & Augmentation]
        B -->|Train / Val / Test Split| C[PyTorch DataLoader<br/>Batch Size: 64]
    end

    subgraph Model_Architecture [2. ResNet-18 Architektur]
        C --> D[Input Layer<br/>3x150x150 RGB]
        D --> E[Initial Block<br/>Conv2D 7x7, Stride 2, BN, ReLU, MaxPool]
        
        E --> F[Layer 1: 2x BasicBlock<br/>64 Feature Maps]
        F --> G[Layer 2: 2x BasicBlock<br/>128 Feature Maps]
        G --> H[Layer 3: 2x BasicBlock<br/>256 Feature Maps]
        H --> I[Layer 4: 2x BasicBlock<br/>512 Feature Maps]
        
        I --> J[Global Average Pooling<br/>AdaptiveAvgPool2d]
        J --> K[Fully Connected Layer<br/>Linear 512 -> 6]
    end

    subgraph Training_Optimization [3. Training & Optimierung]
        K --> L[Softmax / Cross-Entropy Loss]
        L --> M[Optimizer: Adam<br/>LR: 1e-3 | Weight Decay: 1e-4]
        M --> N[LR-Scheduler: StepLR<br/>Faktor 0.5 alle 10 Epochen]
        N -.->|Gewichte-Update| Model_Architecture
    end

    subgraph Output [4. Evaluation & Inferenz]
        K --> O[Klassifikation<br/>6 Klassen]
        O --> P[Klassen: buildings, forest, glacier,<br/>mountain, sea, street]
    end