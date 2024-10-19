# Main Deep Learning Fields and Their Most Used Models

* Computer Vision
    * Subfields
        * Image Classification
            * CNN-based Classification
                * ResNet
                * DenseNet
                * EfficientNet
        * Image Segmentation
            * Semantic Segmentation
                * DeepLab (Atrous Spatial Pyramid Pooling)
                * FCNs (Fully Convolutional Networks)
            * Instance Segmentation
                * Mask R-CNN
                * SOLO (Segmentating Objects by Locations)
            * Panoptic Segmentation
                * Panoptic-DeepLab
                * UPSNet
        * Object Detection
            * Single-stage Detectors
                * YOLO (v3, v4, v5, v8)
                * RetinaNet (Focal Loss)
            * Two-stage Detectors
                * Faster R-CNN
                * R-FCN (Region-based Fully Convolutional Networks)
            * Anchor-free Methods
                * CenterNet
                * FCOS (Fully Convolutional One-Stage Object Detection)
        * 3D Vision & Depth Estimation
            * 3D Object Detection
                * PointNet
                * PointRCNN
            * 3D Scene Representation
                * NeRF (Neural Radiance Fields)
            * Depth Estimation
                * MiDaS (Monocular Depth Estimation)
                * GANet (Stereo-based Depth Estimation)
        * OCR (Optical Character Recognition)
            * Deep Learning-Based OCR
                * CRNN (Convolutional Recurrent Neural Network)
                * ASTER (Attention-based Recognition)
                * EAST
                * CRAFT
        * Image Enhancement
            * Super-Resolution
                * SRCNN (Super-Resolution CNN)
                * EDSR (Enhanced Deep Super-Resolution)
            * Image Denoising
                * Denoising Autoencoders
                * DnCNN (Residual Learning of Deep CNNs)
            * Image Restoration
                * DeRainNet, DeblurGAN
        * Data Augmentation
            * Deep Learning-Based Techniques
                * GANs (Generative Data Augmentation)
                * StyleGAN
                * AutoAugment (Search-based Augmentation)
            * Adversarial Training
                * FGSM
                * PGD
        * Image & Text
            * Vision-Language Models
                * CLIP (Contrastive Learning with Image-Text Pairs)
                * BLIP (Bootstrapping Language-Image Pre-training)
            * Image Captioning
                * Show, Attend and Tell (Attention-based)
                * Transformer-based Models (Image Transformer)
            * Visual Question Answering (VQA)
                * ViLBERT (Vision-and-Language BERT)
                * LXMERT
        * Video Processing
            * Action Recognition
                * 3D CNNs (I3D, C3D)
                * SlowFast Networks (for Efficient Video Processing)
            * Video Object Detection & Tracking
                * DeepSORT (Object Tracking)
                * SiamMask (Real-time Tracking)
            * Video Anomaly Detection
                * Autoencoders, LSTM-CNN Hybrids
                * Spatiotemporal CNNs (ST-CNN)
        * Image Generation
            * GAN-based Generation
                * StyleGAN, BigGAN
                * CycleGAN (Unpaired Image-to-Image Translation)
            * Diffusion Models
                * Denoising Diffusion Probabilistic Models (DDPM)
            * Neural Style Transfer
                * Adaptive Instance Normalization (AdaIN)
                * Deep Photo Style Transfer
        * Vision Models Interpretability
            * Explainability for Deep Models
                * Grad-CAM, Grad-CAM++
                * SHAP (Shapley Additive Explanations)
                * Integrated Gradients
        * Image Similarity Search
            * Siamese Networks
                * Deep Metric Learning (Contrastive Loss, Triplet Loss)
                * Learning Embeddings for Image Retrieval (DeepRank)
        * Self-Supervised Learning (SSL)
            * Representation Learning
                * SimCLR (Simple Framework for Contrastive Learning)
                * MoCo (Momentum Contrast)
                * BYOL (Bootstrap Your Own Latent)
        * Performance Optimization
            * Model Quantization
                * Post-training Quantization
                * Quantization-aware Training (QAT)
            * Knowledge Distillation
                * Teacher-Student Models (Distilling CNNs and ViTs)
                * TinyML (Efficient Deployment of Small Models)
            * Pruning
                * Structured and Unstructured Pruning (Lottery Ticket Hypothesis)
    * Models
        * Convolutional Neural Networks (CNNs)
            * Classic CNN Architectures
                * LeNet
                * AlexNet
                * VGG
            * Modern Architectures
                * ResNet (Residual Connections)
                * DenseNet (Dense Connections)
                * EfficientNet (Scaling Efficiently)
                * MobileNet (Lightweight Models for Mobile)
        * Vision Transformers (ViTs)
            * Pure Transformer Models
                * ViT (Vision Transformer)
                * DeiT (Data-Efficient Transformers)
            * Hybrid Models (CNN-Transformer)
                * Swin Transformer (Hierarchical Vision Transformers)
                * PiT (Pooling-based Vision Transformer)
        * Generative Adversarial Networks (GANs)
            * Unconditional GANs
                * DCGAN (Deep Convolutional GAN)
                * StyleGAN, StyleGAN2
            * Conditional GANs
                * pix2pix (Image-to-Image Translation)
                * GauGAN, BicycleGAN (for image synthesis)
        * Object Detection Models
            * YOLO Family (Real-time Object Detection)
                * YOLOv3, YOLOv4, YOLOv5, YOLOv8
            * Faster R-CNN
                * RPNs (Region Proposal Networks)
                * FPN (Feature Pyramid Networks)
            * Anchor-free Detectors
                * FCOS, CenterNet
        * Image Segmentation Models
            * U-Net
                * U-Net++ (Enhanced U-Net for Medical Imaging)
            * DeepLab Family
                * DeepLabV3, DeepLabV3+ (with Atrous Convolutions)
            * Mask R-CNN
                * Instance Segmentation with Region Proposal Networks
        * Other Vision Models
            * Autoencoders
                * Denoising Autoencoders
                * Variational Autoencoders (VAEs) for Image Generation
            * Attention-based Models
                * DETR (End-to-End Object Detection using Transformers)
                * Perceiver (Generalized Transformers for Multimodal Data)
---
* Natural Language Processing
    * Subfields
        * Text Classification
            * Traditional Text Classification (Sentiment Analysis, Spam Detection)
            * Hierarchical Text Classification (Multiple levels of classification)
            * Multi-Label Classification (Assigning multiple labels to a single text)
            * Zero-Shot Classification (Using models like GPT-3, TARS for unseen categories)
            * Text Toxicity Detection (BERT-based models, RoBERTa for hate speech)
        * Machine Translation
            * Neural Machine Translation (NMT)
                * Transformer-based NMT (Google's Transformer, MarianMT)
            * Multilingual Models
                * mBART, M2M-100 (Multilingual to Multilingual translation)
            * Low-resource Translation
                * Zero-shot, Few-shot translation with large pre-trained models (mT5)
            * Unsupervised Machine Translation
                * Monolingual Data Translation (XLM, MASS)
        * Textual Entailment Prediction
            * Natural Language Inference (NLI)
            *   Pre-trained models fine-tuned for entailment (BERT, RoBERTa, DeBERTa)
            * Multi-Task Learning for NLI
                * Models trained jointly on multiple tasks like MNLI, RTE, SNLI datasets
        * Named Entity Recognition
            * Entity Extraction
                * BERT-based Fine-tuning for Named Entities (BERT-NER, Flair)
            * Biomedical NER
                * SciBERT, BioBERT (for domain-specific NER)
            * Multilingual NER
                * XLM-R (Cross-lingual NER)
        * Seq2Seq (Sequence-To-Sequence)
            * Attention Mechanisms
                * Seq2Seq models with Bahdanau/Luong Attention (for machine translation, summarization)
            * Transformers for Seq2Seq
                * T5 (Text-to-Text Transfer Transformer)
                * BART (Bidirectional and Auto-Regressive Transformers)
            * Text Summarization
                * Abstractive Summarization (T5, PEGASUS)
                * Extractive Summarization (BERTSUM, MatchSum)
        * Text Similarity Search
            * Sentence Embedding Models
                * Sentence-BERT (S-BERT), Universal Sentence Encoder (USE)
            * Semantic Search
                * Dense Passage Retrieval (DPR)
                * ColBERT (Efficient BERT-based Retrieval)
            * Cross-encoder vs. Bi-encoder approaches
                * Efficient vs. accurate text similarity measures using pre-trained language models
        * Language Modeling
            * Causal Language Modeling
                * GPT (Generative Pre-training)
            * Masked Language Modeling (MLM)
                * BERT-style Models (Bidirectional Encoding)
            * Autoregressive vs. Autoencoding
                * GPT (autoregressive) vs. BERT (autoencoding) paradigms
        * Parameter Efficient Fine-Tuning
            * Adapters
                * Efficient Fine-tuning with Adapter Layers (ALBERT, Adapter-BERT)
            * Prompt-based Learning
                * Few-shot prompting (GPT-3-style prompts, P-Tuning)
            * LoRA (Low-Rank Adaptation)
                * Low-resource fine-tuning techniques
            * BitFit
                * Fine-tuning bias terms only in large models
        * Other
            * Question Answering (QA)
                * Extractive QA (SQuAD-based, BERT-style models)
                * Generative QA (GPT-3, T5 for open-domain QA)
            * Speech-to-Text and ASR
                * Transformer-based ASR models (Wav2Vec 2.0)
            * Cross-Lingual Understanding (XLU)
                * Multilingual Models for understanding across languages (XLM, XLM-R)
            * Sentiment Analysis
                * Fine-tuned BERT, GPT models for emotion and sentiment prediction
    * Models:
        * Recurrent Neural Networks (RNNs)
            * Vanilla RNNs (Limited in long-term dependencies)
            * Bidirectional RNNs (Context from both directions)
            * Limitations: Vanishing gradients, limited memory for long sequences
        * Long Short-Term Memory Networks (LSTMs)
            * Classic LSTMs
                * Improved memory retention with gating mechanisms
            * Bidirectional LSTMs
                * Better context for text sequences from both directions
            * Attention Mechanisms
                * LSTMs with attention for better sequence modeling (Bahdanau, Luong)
        * Transformers
            * Self-Attention Mechanism
                * Captures long-range dependencies efficiently
            * Encoder-Decoder Architecture
                * Widely used in sequence-to-sequence tasks (NMT, text generation)
            * Scalability: Highly parallelizable, compared to RNNs and LSTMs
            * Popular Transformer Models
                * BERT (Bidirectional Encoder Representations from Transformers)
                    * Masked Language Modeling (MLM)
                    * Variants: BERT-Base, BERT-Large, ALBERT (smaller, faster)
                * GPT (Generative Pre-trained Transformer)
                    * GPT-2, GPT-3 (Causal Language Modeling)
                    * Focus on text generation and completion
                * T5 (Text-to-Text Transfer Transformer)
                    * Unified model for multiple NLP tasks (text generation, translation, QA)
                    * Encoder-decoder architecture for seq2seq tasks
                * BART (Bidirectional and Auto-Regressive Transformer)
                    * Combines both BERT (encoder) and GPT (decoder) traits
        * Sequence-to-Sequence Models
            * Classic Seq2Seq with Attention
                * Encoder-decoder structure with LSTMs, GRUs, or Transformers
            * Transformers for Seq2Seq Tasks
                * T5, BART, MarianMT for machine translation, summarization, text generation
        * Pre-trained Language Models (PLMs)
            * BERT Variants
                * RoBERTa (Robustly Optimized BERT)
                    * Pre-trained on a larger corpus with better optimization techniques
                * DistilBERT
                    * Smaller, faster, and lighter version of BERT
                * ALBERT
                    * Parameter-efficient version of BERT using factorized embedding layers
                * DeBERTa (Decoding-enhanced BERT)
                    * Uses disentangled attention mechanisms for better encoding
            * Multilingual Pre-trained Models
                * XLM-R (Cross-lingual Language Model)
                    * Pre-trained on multiple languages for cross-lingual tasks
                * mBERT (Multilingual BERT)
                    * Pre-trained for over 100 languages
---
* Structured Data
    * Subfields
        * Structured Data Classification
            * Binary and Multi-class Classification (using tabular data)
            * Multi-Label Classification (handling multiple output labels)
            * Anomaly Detection (Deep learning for fraud detection, outliers)
            * Time Series Classification (classifying sequential structured data)
        * Recommendation
            * Collaborative Filtering (Matrix factorization with neural networks)
            * Deep Learning-based Recommendations (Neural Collaborative Filtering, NCF)
            * Contextual Recommendations (using features like time, user behavior, etc.)
            * Hybrid Models (combining content-based and collaborative approaches with deep learning)
        * Regression (for continuous data predictions)
            * Deep Regression (using deep networks for continuous output prediction)
            * Feature Interaction Modeling (deep networks learning complex feature interactions)
        * Data Imputation & Feature Engineering
            * Deep Autoencoders for Missing Data Imputation
            * Neural Networks for Automatic Feature Learning
    * Models:
        * Feedforward Neural Networks (FFNNs)
            * Fully Connected Layers for structured data inputs
            * Multi-layer Perceptrons (MLPs)
        * Decision Trees (with neural network ensembles)
            * Neural Decision Trees (hybrid architectures combining neural networks with decision trees)
            * Neural-Boosting Algorithms (like NODE: Neural Oblivious Decision Ensembles)
        * Autoencoders for Feature Extraction
            * Denoising Autoencoders (for noise-robust feature extraction)
            * Variational Autoencoders (VAEs) for structured data representation
        * Gradient Boosting Machines (GBMs) integrated with neural networks
            * Neural Gradient Boosting (combining gradient boosting with neural networks for tabular data)
            * Gradient Boosting Decision Trees (GBDT) + FFNN Ensembles (using both shallow and deep models)
        * Self-Supervised Learning for Tabular Data
            * Self-supervised Pre-training on structured data (using techniques like contrastive learning)
        * Graph Neural Networks (GNNs) for Structured Data
            * GNNs for relational and network-based structured data
---
* Time Series Analysis
    * Subfields:
        * Time Series Classification
            * Univariate and Multivariate Time Series Classification
            * Pattern Recognition in Sequential Data
        * Anomaly Detection
            * Unsupervised and Semi-supervised Anomaly Detection in Time Series
            * Real-Time Anomaly Detection (with streaming data)
        * Time Series Forecasting
            * Short-term and Long-term Forecasting
            * Multi-step Forecasting (predicting multiple time steps ahead)
            * Probabilistic Forecasting (with uncertainty estimation)
        * Time Series Imputation
            * Missing Data Imputation in Time Series (using deep learning models)
        * Time Series Segmentation
            * Detecting structural changes or segmenting trends in time series data
    * Models:
        * Recurrent Neural Networks (RNNs) and Long Short-Term Memory Networks (LSTMs)
            * Bidirectional LSTMs (for capturing past and future dependencies)
            * Gated Recurrent Units (GRUs) for time series data
        * Temporal Convolutional Networks (TCNs)
            * Dilated Convolutions for Long-Term Dependencies
            * Residual Connections for stable learning in time series
        * Transformer-based Models for Time Series
            * Temporal Fusion Transformers (TFTs) for interpretable forecasting
            * Self-Attention Mechanisms for handling long-range dependencies in time series
        * Prophet (for Forecasting)
            * Deep Learning Hybrids (combining Prophet with deep models for advanced forecasting)
        * DeepAR
            * AutoRegressive Recurrent Neural Networks for probabilistic forecasting
        * N-BEATS
            * Neural Basis Expansion Analysis for interpretable time series forecasting
        * Self-Supervised Learning for Time Series
            * Contrastive Predictive Coding (CPC) for time series representation learning
---
* Generative Deep Learning
    * Subfields
        * Image Generation
            * Unconditional and Conditional Image Generation
            * Super-Resolution (high-resolution image generation from low-resolution input)
            * Image Inpainting (filling missing regions in images)
        * Style Transfer
            * Neural Style Transfer (applying the artistic style of one image to another)
            * Real-time Style Transfer (using lightweight models for faster inference)
        * Text Generation
            * Language Modeling for Text Generation (autoregressive models like GPT)
            * Conditional Text Generation (using prompts or specific inputs to generate text)
            * Story and Dialogue Generation
        * Graph Generation
            * Molecular Graph Generation (creating molecular structures in drug discovery)
            * Social Network Generation (generating graph structures with specific properties)
        * Video Generation
            * Video Prediction (forecasting future frames in a video sequence)
            * Deepfake Generation (synthesizing realistic fake videos)
        * Music Generation
            * Composition and Style Transfer in Music (generating music with specific styles)
        * Data Augmentation
            * Synthetic Data Generation (for enhancing model training with diverse data)
        * Other
            * Text-to-Image Generation (generating images based on text descriptions)
            * Image-to-Text Generation (e.g., automatic image captioning)
    * Models:
        * Generative Adversarial Networks (GANs)
            * DCGANs (Deep Convolutional GANs for generating images)
            * CycleGANs (for unpaired image-to-image translation)
            * StyleGANs (for high-quality, controllable image generation)
            * BigGANs (for generating high-resolution, large-scale images)
            * Wasserstein GANs (WGANs) (for stable GAN training)
        * Variational Autoencoders (VAEs)
            * Beta-VAEs (for disentangled representations in generative tasks)
            * Conditional VAEs (CVAE) (for conditional generation based on input variables)
            * Vector Quantized VAEs (VQ-VAE) (for learning discrete representations)
        * Diffusion Models
            * Denoising Diffusion Probabilistic Models (DDPM) (for iterative generation of images by reversing a noise process)
            * Latent Diffusion Models (efficient diffusion models for high-dimensional data)
        * Autoregressive Models
            * PixelCNN (for generating images pixel by pixel)
            * WaveNet (for generating raw audio waveforms)
        * Flow-based Models
            * Normalizing Flows (for invertible transformations and likelihood-based generation)
        * Energy-Based Models (EBMs)
            * Deep Energy-Based Generative Models (for energy landscape exploration in generative tasks)
---
* Audio Data
    * Subfields:
        * Speech Recognition
            * Automatic Speech Recognition (ASR)
            * End-to-End Speech Recognition (using deep learning models)
            * Speaker Identification and Verification
        * Sound Classification
            * Environmental Sound Classification (classifying non-speech sounds)
            * Music Genre Classification (classifying types of music)
        * Speech Synthesis
            * Text-to-Speech (TTS) synthesis (converting text into speech)
            * Voice Cloning (generating synthetic voices similar to a specific speaker)
        * Audio Event Detection
            * Detecting and classifying events in audio streams (e.g., footsteps, alarms)
        * Audio Enhancement
            * Noise Reduction (removing background noise using deep models)
            * Speech Enhancement (improving speech intelligibility in noisy environments)
        * Music Generation and Processing
            * Melody Generation (generating new melodies using deep learning)
            * Music Style Transfer (changing the style of music)
        * Other
            * Audio Segmentation (dividing audio into meaningful segments)
            * Audio-to-Text Alignment (synchronizing text with audio)
    * Models:
        * WaveNet
            * Deep Generative Models for raw audio waveform synthesis
        * Convolutional Neural Networks (CNNs) for Spectrograms
            * 2D CNNs on Mel-Spectrograms for Audio Classification
            * CNN + RNN Hybrids (for combining spatial and temporal patterns in spectrograms)
        * Recurrent Neural Networks (RNNs)
            * Long Short-Term Memory Networks (LSTMs) for temporal audio sequence modeling
            * Gated Recurrent Units (GRUs) for more efficient sequence learning in audio
        * Transformer Models for Audio Processing
            * Music Transformer (for melody and music sequence generation)
            * Speech Transformer (for end-to-end speech recognition)
            * AST (Audio Spectrogram Transformer) (for sound classification and event detection)
        * Self-Supervised Audio Models
            * Wav2Vec (for pre-training on raw audio and fine-tuning on downstream tasks)
            * HuBERT (Hidden-Unit BERT for self-supervised learning of speech representations)
        * Autoencoders for Audio
            * Denoising Autoencoders for noise reduction and enhancement
            * Variational Autoencoders (VAEs) for generating new audio samples
        * Diffusion Models for Audio
            * DiffWave (for generating high-quality speech from text or other audio inputs)
---
* Reinforcement Learning
    * Subfields:
        * Model-Free Reinforcement Learning
            * Value-Based Methods (e.g., Q-Learning, DQN)
            * Policy-Based Methods (e.g., Policy Gradient, Actor-Critic methods)
        * Model-Based Reinforcement Learning
            * Learning environment models to plan or simulate future states
            * World Models (using learned representations of environments for planning)
        * Multi-Agent Reinforcement Learning (MARL)
            * Collaboration and competition between multiple agents
            * Self-play and emergent behaviors
        * Hierarchical Reinforcement Learning
            * Learning policies at multiple levels of abstraction
        * Meta-Reinforcement Learning
            * Learning how to learn more efficiently across tasks
        * Offline Reinforcement Learning
            * Learning policies from a fixed dataset of interactions
        * Continuous Control
            * Learning to control physical systems (e.g., robotics, autonomous driving)
        * Other
            * Reward Shaping (guiding learning through carefully designed rewards)
            * Exploration-Exploitation Balance (e.g., curiosity-driven exploration)
    * Models:
        * Q-Learning
            * Basic Q-Learning (for discrete action spaces)
            * Double Q-Learning (to address overestimation bias)
        * Deep Q-Networks (DQN)
            * Dueling DQN (to better separate value estimation and action selection)
            * Double DQN (for more stable Q-value updates)
            * Rainbow DQN (combining multiple DQN improvements: Double DQN, Dueling DQN, etc.)
        * Policy Gradient Methods
            * Advantage Actor-Critic (A3C) (for parallelized policy learning)
            * Proximal Policy Optimization (PPO) (for stable, efficient policy updates)
            * Trust Region Policy Optimization (TRPO) (constraining policy updates for improved stability)
        * Soft Actor-Critic (SAC)
            * Maximum Entropy Reinforcement Learning (optimizing both reward and policy entropy)
            * Continuous Action Spaces (SAC is designed for tasks with continuous action spaces)
        * Actor-Critic Methods
            * Deep Deterministic Policy Gradient (DDPG) (for continuous control tasks)
            * Twin Delayed DDPG (TD3) (addressing overestimation in DDPG)
        * Model-Based RL Methods
            * MuZero (learning a model of the environment without knowing the rules in advance)
            * Dreamer (using learned world models for long-horizon planning)
        * Evolutionary Strategies
            * Genetic Algorithms (for evolving neural network policies)
            * Neuroevolution (for optimizing neural network architectures through evolution)
        * Self-Supervised Learning in RL
            * Contrastive Methods (learning representations of states for better policy learning)
        * Deep Reinforcement Learning in Games
            * AlphaGo, AlphaZero (for board games like Go, Chess)
            * OpenAI Five (for multi-agent, cooperative games like Dota 2)
---
* Graph Data
    * Subfields:
        * Node Classification
            * Classifying individual nodes within a graph (e.g., social network analysis)
        * Link Prediction
            * Predicting the likelihood of edges forming between nodes (e.g., recommendation systems)
        * Graph Classification
            * Classifying entire graphs (e.g., molecular graph classification)
        * Graph Generation
            * Generating new graphs based on learned distributions (e.g., molecular synthesis)
        * Graph Clustering
            * Grouping nodes into clusters based on similarity (e.g., community detection)
        * Graph Embedding
            * Learning low-dimensional representations of nodes or graphs for downstream tasks
        * Temporal Graphs
            * Analyzing graphs with time-evolving structures (e.g., dynamic social networks)
        * Heterogeneous Graphs
            * Handling graphs with different types of nodes and edges (e.g., multi-relational data)
        * Other
            * Graph-based Reinforcement Learning (using graph structures for RL tasks)
            * Explainability in Graph Neural Networks (interpreting GNN outputs)

    * Models:
        * Graph Neural Networks (GNNs)
            * General framework for learning on graph-structured data
        * Graph Convolutional Networks (GCNs)
            * Spectral-based methods for semi-supervised learning on graphs
            * ChebNet (using Chebyshev polynomials for efficient convolutions on graphs)
        * Graph Attention Networks (GATs)
            * Attention mechanisms for learning node importance in GCNs
        * Message Passing Neural Networks (MPNNs)
            * Framework for message passing among nodes for information aggregation
        * GraphSAGE
            * Sample and aggregate methods for inductive learning on large graphs
        * Spatial-Temporal Graph Convolutional Networks (ST-GCN)
            * Handling spatial and temporal dynamics in graph data (e.g., traffic prediction)
        * Relational Graph Convolutional Networks (R-GCNs)
            * Extending GCNs to handle heterogeneous graphs with multiple relation types
        * Diffusion Convolutional Neural Networks (DCNNs)
            * Leveraging diffusion processes for learning on graph-structured data
        * Graph Variational Autoencoders (GVAE)
            * Generative models for learning latent representations of graph structures
        * Graph Transformers
            * Transformers adapted for graph data (e.g., Graphormer)
        * Graph Reinforcement Learning
            * Combining GNNs with RL techniques for decision-making on graphs
