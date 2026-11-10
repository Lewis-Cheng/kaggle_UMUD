**Computer Assisted Surgery** 



**ISSN: 2469-9322 (Online) Journal homepage: www.tandfonline.com/journals/icsu21** 



# **Reinforcement learning for medical image analysis: a systematic review of algorithms, engineering challenges, and clinical deployment** 

**Masuda Begum Sampa, Nor Hidayati Abdul Aziz, Md. Siddikur Rahman, Nor Azlina Ab. Aziz, Rosli Besar & Anith Khairunnisa Ghazali** 

**To cite this article:** Masuda Begum Sampa, Nor Hidayati Abdul Aziz, Md. Siddikur Rahman, Nor Azlina Ab. Aziz, Rosli Besar & Anith Khairunnisa Ghazali (2026) Reinforcement learning for medical image analysis: a systematic review of algorithms, engineering challenges, and clinical deployment, Computer Assisted Surgery, 31:1, 2597553, DOI: 10.1080/24699322.2025.2597553 

**To link to this article:** <u>https://doi.org/10.1080/24699322.2025.2597553</u> 













© 2025 Multimedia University. Published by Informa UK Limited, trading as Taylor & Francis Group. 

Published online: 26 Dec 2025. 



Submit your article to this journal 

Article views: 121 



View related articles 



View Crossmark data 

Full Terms & Conditions of access and use can be found at https://www.tandfonline.com/action/journalInformation?journalCode=icsu21 

Computer Assisted surgery 2026, VoL. 31, No. 1, 2597553 https://doi.org/10.1080/24699322.2025.2597553 

#### Review ARticle 



## **Reinforcement learning for medical image analysis: a systematic review of algorithms, engineering challenges, and clinical deployment** 

Masuda Begum Sampa<sup>a,b</sup> , Nor Hidayati Abdul Aziz<sup>a</sup> , Md. Siddikur Rahman<sup>c</sup> , Nor Azlina Ab. Aziz<sup>a</sup> , Rosli Besar<sup>a</sup> and Anith Khairunnisa Ghazali<sup>a</sup> 

aFaculty of engineering & technology, multimedia university, melaka, malaysia; buniversity of science and technology Chittagong (ustC), Chattogram, Bangladesh;<sup>c</sup> department of statistics, Begum rokeya university, rangpur, Bangladesh 

##### **ABSTRACT** 

Reinforcement learning (Rl) has emerged as a powerful artificial intelligence paradigm in medical image analysis, excelling in complex decision-making tasks. this systematic review synthesizes the applications of Rl across diverse imaging domains—including landmark detection, image segmentation, lesion identification, disease diagnosis, and image registration—by analyzing 20 peer-reviewed studies published between 2019 and 2023. Rl methods are categorized into classical and deep reinforcement learning (DRl) approaches, focusing on their performance, integration with other machine learning models, and clinical utility. Deep Q-Networks (DQN) demonstrated strong performance in anatomical landmark detection and cardiovascular risk estimation, while Proximal Policy Optimization (PPO) and Advantage Actor-critic (A2c) achieved optimal policy learning for vessel tracking. Policy gradient methods such as ReiNFORce, twin-Delayed Deep Deterministic Policy Gradient (tD3), and Soft Actor-critic (SAc) were successfully applied to breast lesion detection, white-matter connectivity analysis, and vertebral segmentation.Monte carlo learning, meta-Rl, and A3c methods proved effective for adaptive questioning, image quality evaluation, and multimodal image registration. to consolidate these findings, we propose a unified Reinforcement learning Medical imaging (RlMi) framework encompassing four core components: state representation, policy optimization, reward formulation, and environment modeling. this framework enhances sequential agent learning, stabilizes navigation, and generalizes across imaging modalities and tasks. Key challenges remain, including optimizing task-specific policies, integrating anatomical contexts, addressing data scarcity, and improving interpretability. this review highlights Rl’s potential to enhance accuracy, adaptability, and efficiency in medical image analysis, providing valuable guidance for researchers and clinicians applying Rl in real-world healthcare settings. 

##### **KEYWORDS** 

Medical image analysis; reinforcement learning; systematic literature review 

### **1. Introduction** 

Medical image analysis is important in healthcare as it enhances disease diagnosis, treatment planning, and monitoring supports medical research, education, telemedicine, and the advancement of personalized medicine. Due to significant advancements in artificial intelligence (Ai) technology in recent years, various application fields have increasingly adopted Ai [1]. with ongoing technological advancements and the integration of Ai, specifically machine learning (Ml) techniques, medical imaging in healthcare continues to evolve and contributes improving patient care [2]. 

> **CONTACT** Nor Hidayati Abdul Aziz hidayati.aziz@mmu.edu.my Faculty of engineering & technology, multimedia university, 75450 melaka, malaysia. 

> © 2025 multimedia university. published by informa uK Limited, trading as taylor & Francis group. 

> this is an open Access article distributed under the terms of the Creative Commons Attribution License (http://creativecommons.org/licenses/by/4.0/), which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited. the terms on which this article has been published allow the posting of the Accepted manuscript in a repository by the author(s) or with their consent. 

2 M. B. SAMPA et Al. 

#### **_1.1. Machine learning paradigms in medical imaging_** 

Machine learning (Ml) in medical imaging generally comprises supervised, unsupervised, semi-supervised, and reinforcement learning (Rl) paradigms (shown in Figure 1) [3]. Supervised learning uses labeled datasets to train predictive models [4] but depends heavily on large annotated datasets. Unsupervised learning operates on unlabeled data, extracting patterns or clusters [4], and semi-supervised learning combines both labeled and unlabeled data to improve model performance [5]. 

Reinforcement learning differs from these paradigms by training an agent to interact with an environment to maximize cumulative rewards [6]. Rl problems are characterized by closed-loop decision-making, absence of a guiding tutor, and actions influencing both immediate and long-term outcomes [7,8]. Rl is well-suited for dynamic, sequential tasks such as robotics, game strategy, and autonomous systems [9–11]. 

in medical imaging, Rl’s adaptability makes it valuable for object detection, segmentation, disease diagnosis, and landmark localization, especially in scenarios with limited labeled data or dynamic contexts. For instance, Rl can refine segmentation performance _via_ iterative feedback, optimize long-term clinical decision-making in treatment planning, and handle complex, multi-step diagnostic workflows. Notable contributions include image segmentation [12–14], landmark localization [15], and computational model personalization [16,17]. 

Several prior reviews have examined Rl and deep Rl (DRl) applications in medical imaging from various perspectives [2,7,18–23]. while these works discuss principles, applications, and challenges, none provide a PRiSMA-compliant systematic review focused exclusively on Rl-based medical image analysis (RlMi) with a structured methodology for selecting and analyzing studies. this gap motivates our work. 

#### **_1.2. Reinforcement learning in medical image analysis_** 

Rl in medical imaging has demonstrated promising results across multiple tasks: 

landmark Detection – Agents navigate within 2D/3D images to localize anatomical points [15,24,25]. image Segmentation – iterative refinement of boundaries using environment feedback improves accuracy, even with modality or anatomy variations [26]. Disease Diagnosis – Optimizing long-term treatment strategies or integrating multimodal information [27,28]. Registration and Synthesis – Aligning multimodal scans [29] and synthesizing enhanced images from non-enhanced ones [30]. Other Applications – image quality assessment [31], visual question answering [32], and vessel tree reconstruction [33]. 



<!-- Start of picture text -->
Al<br>ML<br>oo}® Y<br>. e 2 uy,<br>7 o GY<br>g 2 X%<br>> B “o<br>€<br>a<br><!-- End of picture text -->

**Figure 1.** A Venn diagram showing relationship between Ai, mL, supervised learning, semi-supervised learning, unsupervised learning, rL, and deep learning. 

cOMPUteR ASSiSteD SURGeRy 3 

these examples illustrate Rl’s potential to address challenges such as variability in anatomy, scarcity of annotations, and the need for adaptive decision-making in medical imaging workflows. 

#### **_1.3. Aims and objectives_** 

the aims of this study are to: 

1. Systematically review the state-of-the-art applications of Rl in medical image analysis. 

2. identify and classify Rl and DRl algorithms used in the field. 

3. Propose a standardized RlMi framework incorporating state representation, policy optimization, reward formulation, and environment design. 

4. Analyze key trends, challenges, and open research directions to advance Rl in clinical imaging contexts. 

the use of Rl in medical image analysis is becoming a trend, as illustrated in Figure 2. Figure 3 shows publications details for 20 reinforcement learning medical imaging (RlMis)articles that were reviewed in this study. 

the growing interest in reinforcement learning-based medical image analysis (RlMi) motivated us to prepare this paper to provide a comprehensive overview of the state-of-the-art RlMi. 

### **2. Basics of reinforcement learning** 

Reinforcement learning (Rl) is an Ml paradigm that trains agents to make sequences of decisions in an environment to maximize a cumulative reward [34]. Reinforcement learning allows one to solve problems _via_ a sophisticated trial and error process by placing a learning agent in an environment and providing a reward signal based on actions performed. the agent then adapts its strategy to maximize its expected cumulative reward. the four main components of Rl are environment, state, agent, and reward (Figure 4). Reinforcement learning algorithms, including deep reinforcement learning (DRl) which combines deep learning’s perceptual understanding with reinforcement learning’s decision-making, have been applied to gaming, control, and increasingly to complex image processing challenges in medical imaging (Figure 5) [18,21,35,36]. 



<!-- Start of picture text -->
By<br>6<br>8<br>2|<br>g<br>5,2 |<br>Ss ||<br>5<br>2<br>2| |<br>0 UJ J i |<br>Landmark Object Disease Image Image Lesion<br>detection localization diagnosis Classification segmentation detection<br>Application Areas<br><!-- End of picture text -->

**Figure 2.** different medical image analysis tasks using rL versus number of papers. 

4 M. B. SAMPA et Al. 



<!-- Start of picture text -->
30<br>25<br>2 20<br>4<br>é<br>1S,<br>a)s<br>:°5 | | fs — A ;<br>2023 2022 2021 2000 a m Journal m Conference = Preprint<br>mRL m DRL<br>(a) (b)<br><!-- End of picture text -->

**Figure 3.** publications details for 20 rLmi articles that were surveyed. (a) distribution of rLmis publications per year (until september 2023), divided into categories according to rL and drL techniques(b) the percentage of surveyed rLmis that are published in various formats, such as journal articles, conference proceedings, and preprints. 



**Figure 4.** Four main components of rL. 

#### **_2.1. RL based problem formulation_** 

in this section, we present Rl-based medical imaging; i.e. methods that use reinforcement learning for policy optimization. every Rl-based problem has four elements in common, which are: (1) State representation, (2) Policy Optimization, (3) Reward Formulation, and (4) environment Building. Following Figure 6. depicts this framework. in the following, we explain each component. 

### **3. Material and methods** 

#### **_3.1. Systematic literature review_** 

this study follows the standard guidelines for systematic literature review (SlR) by Kitchenham and charters [37]. there are three primary stages to the SlR process: (1) Planning the review (2) conducting the review and (3) Reporting the review (Figure 7). 

cOMPUteR ASSiSteD SURGeRy 5 



<!-- Start of picture text -->
Action<br>Environment Reward Experience Agent<br>(task) 7 Function _ Replay 7 Network<br><!-- End of picture text -->

**Figure 5.** the principle of drL [36]. 



**Figure 6.** Components of a reinforcement learning based framework. 



<!-- Start of picture text -->
Planning Conducting Reporting the<br>Review the Review Review<br>Research<br>Questions<br>Search<br>Strategy<br>Data<br>Extraction<br><!-- End of picture text -->

**Figure 7.** phases of sLr. 

6 M. B. SAMPA et Al. 

#### **_3.2. Planning the review_** 

the review procedure consists of (a) the research questions; (b) the search strategy; (c) the inclusion and exclusion criteria (study selection criteria) and ( _d_ ) the data extraction and synthesis of results. 

#### **_3.2.1. Research questions (RQ)_** 

the following questions were formulated to guide the SlR study: 

RQ1: How Rl is used in the research related to medical image analysis? 

RQ2: what are the algorithms used? 

RQ3: what is the future direction? 

#### **_3.2.2. Search strategy_** 

the search strategy for this review involved querying electronic databases, specifically ieee Xplore, Science Direct, and Springer links, for peer-reviewed papers published between January 1st, 2019, and September 30th, 2023, that applied reinforcement learning in medical imaging. the search utilized keywords ‘Reinforcement learning’ and ‘medical image analysis’, combined using Boolean operators ‘AND’ and ‘OR’ to enhance search flexibility, while excluding certain terms with ‘NOt’, as recommended [38,39]. 

1. ‘reinforcement learning’ OR ‘medical image analysis’ 

2. ‘reinforcement learning’, AND ‘medical image analysis’ 

3. ‘reinforcement learning’ NOt ‘game’. 

#### **_3.2.3. Study selection criteria_** 

considering the research questions and objectives of the SlR, specific criteria for inclusion and exclusion were established and adhered to in the retrieved published papers. 

A. **Inclusion criteria.** Primary studies that contain the following will be included: 

- english peer-reviewed studies that provide answers to our research questions. 

- Studies that focus on medical image analysis research 

- Studies that are published between 2019–2023 inclusive. 

B. **Exclusion criteria.** the studies that contain the following will be excluded: 

- Studies that are not in english 

- Studies that are not related to the research questions 

- Duplicated studies 

- Studies before 2019 

#### **_3.2.4. Strategies for data extraction and synthesis_** 

A standardized form that included the necessary data to be extracted—publication details, titles, authors, and publication years—was developed to make the process of extracting and synthesizing data easier. the fields were provided for the reinforcement learning application area, application task, Rl algorithms used, Rl algorithm performance, datasets and number of subjects/data used. these data correspond to the research questions and consequently helped in answering the research questions as well (shown in table 1). 

#### **_3.3. Conducting the review_** 

this section covers the procedure used to find publications for this systematic literature review. the databases were searched using the search strings in order to find possible primary studies for this SlR. the PRiSMA flowchart Figure 8. clearly illustrates the stages followed to obtain our final papers. initially, 137 studies that were not duplicates were chosen from three databases. 137 articles were retrieved in 

cOMPUteR ASSiSteD SURGeRy 7 

|performance<br>Limitations/ Future directions<br>the deep statistical shape<br>model (deepssm) and<br>multitarget reinforcement<br>learning agent (deep<br>multi-agent Q learning<br>network) combined in the<br>suggested framework<br>demonstrated excellent<br>robustness on landmarks<br>with a range of detection<br>difficulties.<br>For improving the accuracy and<br>robustness the method can be<br>modified across various<br>scenarios.<br>Choosing the best dQN<br>architecture varies<br>depending on<br>each landmark, whereas<br>the ideal dQN<br>architecture depends on<br>the environment in which<br>it is implemented.<br>explore intrinsic geometry for<br>improved rL performance,<br>investigate communication<br>strategies among agents,<br>emphasizing knowledge<br>sharing, emphasizing the<br>importance of sharing acquired<br>knowledge, Consider involving<br>human experts in teaching<br>artificial agents actively,<br>emphasize agents learning<br>from both self-play experiences<br>and interaction with trained<br>operators.<br>reinforcement learning<br>(rL)-based landmark<br>identification using a<br>shape model performs<br>better than the suggested<br>rL-based framework<br>without a shape model.<br>to enhance the collection of<br>uncommon anomalies and<br>validate the framework on<br>patients who are younger than<br>eighteen, the database can be<br>expanded. Algorithmic<br>improvements can also be<br>made.<br>(_Continued_)|
|---|
|<br>policy optimization/ rL<br>Algorithms used<br>Combination of rL/<br>drL and other<br>models<br> <br> <br> <br>deep multi-agent<br>Q-learning<br>(deepmaQ) network.<br>deep multi-agent<br>Q-learning<br>(deepmaQ)<br>network + deep<br>statistical shape<br>model (deepssm).<br>ac<br> <br>deep Q network (dQN)<br>(deep Q network<br>(dQN))<br>deep CNN + rL<br>deep Q-network<br>(deepQN), deep<br>deterministic policy<br>gradient (ddpg),<br>twin delayed deep<br>deterministic policy<br>gradient (td3) and<br>actor-critic policy<br>gradient (A2C)<br>graphical lasso and<br>morris sensitivity<br>analysis with deep<br>neural networks<br>(CNN) using rL.|
|data set (No. of<br>subjects/ data)<br> <br>99 dXA, 45 Lge<br>Cmr sequences<br>cardiac mri<br>dataset, 54<br>whole-head Ct<br>images, and 41<br>upper-half head<br>Ct images.<br>72 fetal head us<br>scans, 455<br>short-axis cardi<br>mr images, 832<br>isotropic 1 mm<br>mri scans<br> <br>260 patients.|
|task<br>references<br>modality<br>multiple landmark<br>detection<br>[25]<br>Body dual-energy<br>X-ray<br>absorptiometry<br>(dXA), cardiac<br>mri, and head<br>Ct images<br>Automatic detection<br>of anatomical<br>landmarks<br>[24]<br>ultrasound (us),<br>and magnetic<br>resonance<br>imaging (mri).<br>detecting and<br>quantifying pelvic<br>abnormalities<br>[40]<br>magnetic<br>resonance (mr)<br>images|
|Application area<br>Landmark<br>detection<br>Landmark<br>detection<br>Landmark<br>detection.|



8 M. B. SAMPA et Al. 

|Limitations/ Future directions<br> <br>it can be used to expand the<br>suggested framework for<br>learning-to-learn to more<br>widely-used landmark-detection<br>tasks.<br>subsequent research endeavors<br>could integrate the carotid<br>vasomotion’s periodicity into<br>the computation procedure<br>and investigate the anatomic<br>connections among<br>neighboring frames.<br>(_Continued_)|
|---|
|performance<br>the suggested approach<br>delivers an accuracy<br>improvement of 20–30%<br>on average over the<br>chosen reference<br>solutions and is resistant<br>to outliers. the method’s<br>speed is faster, achieving<br>real-time performance on<br>high resolution 3d-Ct<br>volumes. it can also<br>recognize many objects at<br>once and manage the<br>situation where one or<br>more are absent.<br>two diverge datasets prove<br>the feasibility and<br>effectiveness of the<br>proposed learning-to-<br>learn framework.<br>the suggested framework<br>achieves mAes of<br>0.96 ± 0.70 / 0.06 ± 0.04<br>for carotid artery<br>intima-media thickness<br>(Cimt) and 3.02 ± 2.23 /<br>0.18 ± 0.13 for carotid<br>artery lumen diameter<br>(CALd).|
|data set (No. of<br>subjects/ data)<br>policy optimization/ rL<br>Algorithms used<br>Combination of rL/<br>drL and other<br>models<br>the dataset contains<br>1487 3d-Ct<br>volumes from<br>532 patients<br>deep Q-network (dQN)<br>Combines the<br>representational<br>power of modern<br>CNN architectures<br>and cognitive<br>modeling through<br>reinforcement<br>learning (rL).<br>two distinct sets of<br>data. the first<br>one is a<br>400 × 400 fetus<br>midsagittal plane<br>image dataset<br>from in-house<br>prenatal<br>ultrasound tests.<br>the second<br>dataset is a<br>publicly<br>accessible<br>dataset from the<br>2015<br>isBi-sponsored<br>Automatic<br>Cephalometric<br>X-ray Landmark<br>detection<br>Challenge<br>(ACXrLdC).<br>reiNForCe rule<br>Combines u-Net and<br>reinforcement<br>learning (rL).<br>95 patients<br>dQN<br>se-Net + resNeXt|
|references<br>modality<br>n<br> <br>i<br>[41]<br>Ct scan<br>[42]<br>ultrasound and<br>X-ray<br>[28]<br>ultrasound (us)<br>images|
|task<br>real-time detectio<br>of 3d landmark<br>or identification<br>of anatomical<br>structures.<br>Anatomical<br>landmarks<br>detection<br>disease risk<br>estimation.|
|Application area<br>Landmark<br>detection.<br>Landmarks<br>detection<br>disease diagnosis|



cOMPUteR ASSiSteD SURGeRy 9 

|performance<br>Limitations/ Future directions<br>the experimental results<br>demonstrate the superior<br>performance of the<br>suggested method over<br>existing state-of-the-art<br>multi-modal algorithms<br>and other single-modality<br>models.<br>in the future, the current<br>method might be used for<br>other image modalities other<br>than ultrasound images.<br>As opposed to asking every<br>question, the suggested<br>Questioning strategy (Qs)<br>provided a more nuanced<br>assessment by effectively<br>differentiating between<br>those who were<br>reasonable and those<br>who were unreasonable.<br>utilize the framework for a variety<br>of clinical situations outside of<br>dme grading. Create<br>specialized VQA datasets that<br>correspond to different medical<br>tasks for more focused<br>evaluation, to improve the<br>applicability of the framework,<br>incorporate sophisticated<br>image representations, perhaps<br>with the aid of feature<br>extractors. to extend the<br>application of the questioning<br>technique, look into adding<br>open-ended inquiries. examine<br>ways to improve<br>interpretability by including the<br>Questioning strategy (Qs) into<br>the model under evaluation’s<br>training.<br>in comparison to earlier<br>machine learning-based<br>tractography, the<br>suggested drL-based<br>tractography framework<br>shows little performance<br>loss on new, unseen data<br>and competitive results<br>on known data.<br>the framework that is being<br>described is positioned as a<br>useful foundation for upcoming<br>developments in tractography<br>methods. more improved<br>tractography methods can also<br>be developed in the future.<br>(_Continued_)|
|---|
|L<br>Combination of rL/<br>drL and other<br>models<br>CNN + rL<br>reinforcement<br>learning<br> <br>deep reinforcement<br>learning|
|policy optimization/ r<br>Algorithms used<br>reiNForCe rule<br>r<br> <br>monte Carlo (mC)<br>learning and<br>Q-learning.<br>twin-delayed<br>deep-deterministic<br>policy gradient<br>(td3) and and soft<br>Actor-Critic (sAC)|
|data set (No. of<br>subjects/ data)<br>835 patients from<br>the ruijin<br>Hospital.<br>two datasets: the<br>eophtha data<br>set and the<br>indian diabetic<br>retinopathy<br>image dataset<br>(idrid): 148 colo<br>fundus photos of<br>individuals with<br>diabetic<br>retinopathy and<br>62 color fundus<br>photos of<br>individuals with<br>both conditions.<br>simulated FiberCup<br>dataset, and<br>ismrm2015<br>dataset|
|nces<br>modality<br>us images<br>fundus images<br>with dme<br>disease<br>diffusion mri|
|task<br>refere<br>Breast cancer<br>diagnosis.<br>[27]<br>Automatically<br>selecting<br>questions from a<br>pool for an eye<br>image to<br>accurately detect<br>dme grade.<br>[32]<br> <br>developing the<br>tractography<br>process, a<br>technique used<br>in diffusion mri<br>to assess white<br>matter<br>connectivity in<br>the brain<br>accurately and<br>efficiently.<br>[43]|
|Application area<br>disease diagnosis<br>evaluating a VQA<br>system or an<br>automatic<br>adaptive<br>questioning<br>method/<br>disease<br>diagnosis.<br>Assessing the<br>white-matter<br>structural<br>connectivity of<br>a brain/ disease<br>diagnosis.|



10 M. B. SAMPA et Al. 

|<br>performance<br>Limitations/ Future directions<br>the proposed approach has<br>been shown its wide<br>applicability with clinically<br>relevant target tasks of<br>prostate presence<br>detection and gland<br>segmentation, and<br>pneumonia detection.<br>Future research can investigate<br>how the multi-environment<br>formulation uncovers several<br>meanings of task amenability<br>by using non-expert-labeled<br>data. examining the current<br>framework’s adaptability to<br>effectively use expert-labeled<br>data for different target tasks<br>is another possibility.<br>Additionally, the potential<br>benefits of the current<br>technology for applications like<br>active learning can be<br>analyzed, and its suitability for<br>use in resource-constrained<br>clinical settings can be<br>investigated.<br>pAss performs better than<br>the most advanced<br>techniques, producing<br>reconstructions that are<br>more accurate while<br>cutting down on the<br>length of patient<br>assessments by 30% to<br>70%.<br>strategies that choose which of<br>the suggested models is best<br>to utilize based on the patient<br>can be found, and the<br>framework can then be<br>expanded to include other<br>conventional medical imaging<br>devices.<br>Future work can focus on<br>integrating the periodicity<br>of carotid vasomotion<br>into the calculation<br>process and examining<br>the anatomical<br>relationships between<br>adjacent frames.|
|---|
|<br>policy optimization/ rL<br>Algorithms used<br>Combination of rL/<br>drL and other<br>models<br>meta-rL<br>rNN + rL<br> <br>he<br> <br>reiNForCe<br>policy gradient<br>reinforcement<br>learning<br>s.<br>reinforcement learning|
|data set (No. of<br>subjects/ data)<br>image of 259<br>patients<br>5108 visual field<br>samples from<br>139 glaucoma<br>patients and 22<br>healthy<br>individuals in t<br>Netherlands. as<br>well as 1108<br>visual field<br>samples from<br>538 swiss<br>patients.<br>4351 annotated<br>CCA-us frames<br>from 95 patient|
|es<br>modality<br>x-ray, us<br>2d visual image of<br>eye<br>ultrasound images|
|task<br>referenc<br>image Quality<br>Assessment<br>[31]<br>to improve the<br>efficiency of<br>perimetry, a<br>clinical<br>assessment that<br>is used to<br>identify<br>neurological and<br>ocular diseases.<br>[44]<br>to enhance the<br>measurement of<br>carotid lumen<br>diameter (CALd)<br>and intima-media<br>thickness (Cimt)<br>using ultrasound<br>images.<br>[28]|
|Application area<br>Active learning/<br>disease<br>diagnosis<br>disease diagnosis/<br>diagnosing<br>ophthalmic and<br>neuro- logical<br>conditions<br>Cardiovascular<br>Health<br>Assessment/<br>disease<br>diagnosis|



cOMPUteR ASSiSteD SURGeRy 11 

|Limitations/ Future directions<br> <br>rather than using maximal<br>intensity projection (mip)<br>images for training and<br>tracking, they employed the<br>maximum volume Ctp images.<br>the mip appears different from<br>CtA images due to its greater<br>background intensities and<br>noisier look, even if it is more<br>suitable for annotation.<br> <br>in the future, the suggested<br>method’s viability in actual<br>situations can be assessed.<br>Future investigations into the<br>effects of capsule design<br>decisions on algorithm<br>performance, such as coverage<br>tasks utilizing 360-degree<br>camera designs and stereo<br>camera tests for depth<br>estimation, could also be<br>pursued.<br> <br>Future work may include the<br>three-dimensional image<br>registration as well as the<br>generalization to other<br>computer vision problems.<br> <br>the proposed sCrL framework is<br>specifically designed and<br>evaluated on spine mr images.<br>its generalizability to other<br>imaging modalities such as Ct<br>scans or X-rays has not been<br>validated, which limits its<br>broader clinical applicability.|
|---|
|performance<br>in accordance with inter<br>observer agreement, the<br>final vessel tree<br>development had a<br>median recall of 68% and<br>a precision of 70%.<br>in 47.62 s, the pre-trained<br>agent covers 96% of the<br>inspected gastrointestinal<br>instance. this suggests<br>that a strategy for<br>autonomous coverage<br>policy based on deep<br>learning (drL) offers a<br>promising and methodical<br>monitoring potential for<br>the upcoming capsule<br>endoscopes.<br> experiments conducted on<br>paired Ct and mr images<br>of individuals with<br>nasopharyngeal cancer<br>indicate that the<br>current approach reaches<br>cutting-edge results in<br>medical image<br>registration.<br>With an average intersection<br>over union (iou) of 92.3%<br>for detection, a dice<br>coefficient of 92.6% for<br>segmentation, and a<br>mean accuracy of 96.4%<br>for classification, sCrL<br>achieves remarkable<br>performance in both<br>detection and<br>segmentation.|
|<br>Combination of rL/<br>drL and other<br>models<br> <br>Convolutional neural<br>networks (CNN),<br>recurrent neural<br>networks (rNN),<br>and vessel tree<br>construction<br>approaches are<br>combined in the<br>deep<br>reinforcement<br>learning (drL)<br>methodology.<br>deep reinforcement<br>learning<br>A3C-convLstm model <br>sequential<br>Conditional<br>reinforcement<br>Learning network<br>(sCrL) = CNN + rL|
|<br>policy optimization/ rL<br>Algorithms used<br>Advantage actor critic<br>(A2C), combined<br>with proximal policy<br>optimization (ppo).<br>policy gradient method<br>proximal policy<br>optimization (ppo)<br>Actor-critic (A3C).<br>soft Actor-Critic (sAC).|
|data set (No. of<br>subjects/ data)<br>115 subjects<br>825 subjects<br>99 patients<br>240 subjects|
|nces<br>modality<br>3d CtA and Ctp<br>images of the<br>brain<br>computed<br>tomography<br>(Ct) images<br>Ct, mri image<br>mri images|
|refere<br> <br>l<br>[33]<br>[45]<br>is<br>[29]<br>s<br>[36]|
|task<br> <br>extracting the<br>anterior vessel<br>tree in patients<br>with intracrania<br>large vessel<br>occlusion (LVo)<br>Autonomously<br>monitoring<br>organ.<br> <br>Active learning/<br>disease diagnos<br>detecting and<br>segmenting<br>vertebral bodie<br>in medical<br>images|
|Application area<br>extracting the<br>anterior vessel<br>tree<br>Human organ<br>monitoring.<br>image registration<br>image<br>segmentation|



12 M. B. SAMPA et Al. 

|Limitations/ Future directions<br>the existing technique, which is<br>now applicable to single object<br>segmentation, can be applied<br>to multi-object segmentation<br>as well. Additionally, in order<br>to increase training efficiency,<br>a way must be found to<br>simplify the network<br>architecture of the current<br>approach, particularly the<br>ddrL.<br>A wider variety of imaging<br>modalities as input may<br>increase the existing method’s<br>performance.<br>Fast, online algorithms are being<br>developed to provide real-time<br>acquisition advice in future<br>clinics. in future, more<br>sophisticated systems to use<br>expert knowledge to direct<br>operators to concentrate on<br>particular informative regions<br>of the pose space in order to<br>get better results can be<br>developed.<br> <br> <br>potential uses of the suggested<br>framework for automated<br>whole-breast ultrasonography<br>or the analysis of ultrasound<br>recordings of other organs,<br>with modifications made for<br>processing 3d images, could<br>be investigated in future study.|
|---|
|performance<br>Achieves 83.11% dice and<br>85.12% recall in testing<br>data (50 patients) after<br>training on 200 patient<br>data<br>in distinguishing between<br>benign and cancerous<br>tumors, pix-grL achieved<br>an accuracy rate of<br>87.60%<br>the suggested approach<br>produced cutting-edge<br>results in terms of image<br>quality and misalignment<br>resilience.<br> <br>the results demonstrated<br>how well the suggested<br>strategy performed in key<br>frame extraction and<br>video summary compared<br>to other cutting-edge<br>algorithms.|
|L<br>Combination of rL/<br>drL and other<br>models<br>dual-strategy drL<br> <br>graph-driven agent<br>module with<br>context awareness<br>and a dual-level,<br>complementary<br>advantage<br>function based on<br>rewards are<br>combined.<br>CNN + Lstm+ rL<br>the proposed<br>framework<br>integrates<br>reinforcement<br>learning, dual-level<br>filtering, and a<br>group-aware focal<br>loss|
|<br>policy optimization/ r<br>Algorithms used<br>Actor-Critic (AC)<br>method<br>reinforcement learning<br>method<br>policy gradient<br>reiNForCe|
|data set (No. of<br>subjects/ data)<br>200 patients<br>325 subjects<br>166, 3d static us<br>images<br>of 70 subjects<br>653 patients|
|es<br>modality<br>images<br>mri images<br>3d us<br>ultrasound video|
|task<br>referenc<br>Liver tumor<br>segmentation in<br>non-enhanced<br>images without<br>contrast agents.<br>[26]<br>synthesizing<br>gadolinium<br>(gd)-enhanced<br>liver tumors on<br>non-enhanced<br>liver mr images<br>[30]<br>transforms the<br>imaged anatomy<br>to a predefined<br>pose.<br>[46]<br>extracting<br>key-frames from<br>ultrasound (us)<br>videos of breast.<br>[47]|
|Application area<br>image<br>segmentation<br>Liver tumor<br>diagnosis/<br>object<br>localization.<br>pose correction<br>Breast Lesion<br>detection.|



cOMPUteR ASSiSteD SURGeRy 13 

|Limitations/ Future directions|Future work can focus on<br>integrating the periodicity of<br>carotid vasomotion into the<br>calculation process and<br>examining the anatomical<br>relationships between adjacent<br>frames.|
|---|---|
|performance|carotid lumen diameter<br>(CALd) and intima-media<br>thickness (Cimt), the<br>framework achieved a<br>mean absolute error<br>(mAe) of 3.02 ± 2.23<br>pixels/mm and<br>0.18 ± 0.13 mm,<br>respectively. these results<br>surpass the performance<br>of widely adopted<br>methods that rely on<br>larger amounts of<br>annotated data|
|data set (No. of<br>subjects/ data)<br>policy optimization/ rL<br>Algorithms used<br>Combination of rL/<br>drL and other<br>models|4351 annotated<br>CCA-us frames<br>from 95 patients.<br>Q value<br>reinforcement<br>learning|
|task<br>references<br>modality|to enhance the<br>measurement of<br>carotid lumen<br>diameter (CALd)<br>and intima-media<br>thickness (Cimt)<br>using ultrasound<br>images.<br>[28]<br>ultrasound images|
|Application area|Cardiovascular<br>Health<br>Assessment/<br>disease<br>diagnosis|



14 M. B. SAMPA et Al. 



<!-- Start of picture text -->
IEEE Explore Science direct Springer link<br>2019 - 2023 2019 - 2023 2019— 2023<br>2 citation(s) 73 citation(s) 62 citation(s)<br>137 Non-Duplicate<br>Citation Screened<br>Inclusion/Exclusion O Articles Excluded<br>Criteria Applied After Title/Abstract Screen<br>137 Articles Retrieved - -<br>Inclusion/Exclusion 115 Articles Excluded 2 Articles Excluded<br>Criteria Applied After Full Text Screen During Data Extraction<br>20 Articles Included<br><!-- End of picture text -->

**Figure 8.** retrieval, selection, and assessment of literature for inclusion procedures. 

the first phase following title and abstract reviews. two articles were eliminated during the data extraction process, and 115 articles were eliminated when the full contents of the articles were reviewed in the second phase. in the end, this study contained 20 publications for review. 

three reviewers independently screened the studies based on predefined inclusion and exclusion criteria. Any disagreements were resolved through discussion, and when consensus could not be reached, a senior reviewer served as the final arbiter. Data extraction was performed independently by three reviewers using a standardized form, and the extracted information was cross-validated to ensure accuracy and consistency. 

### **4. Results** 

this section reports the SlR results. it involves presenting the findings in the extracted papers and answering the research questions. in this section, we categorized the application of Rl in medical image analysis into different tasks. 

#### **_4.1. Applications of RL in medical image analysis (RQ1)_** 

in medical image analysis, Rl has been used for a wide range of tasks [42], such as landmark detection [24], view plane localization [48], lesion detection [49], and registration [50]. this section synthesizes the application trends and critical insights drawn from 20 systematically reviewed RlMi studies (see table 1 for full summary). to streamline the narrative, we grouped applications into the following categories: 

**Landmark Detection:** landmark points are vital for morphometric analysis and support tasks like segmentation and registration [51,52]. Rl shifts from direct detection to optimal path learning [15]. SGMaRl, a multi-agent Rl framework combining DeepMaQ and DeepSSM, effectively handled incomplete medical images [25]. it achieved robust detection even with 80% missing training data. Zhou GQ et al. [42] proposed a learning-to-learn Rl strategy enhancing multi-landmark localization. Alansary et al. [24] evaluated various DQN variants, showing environment-specific optimality and superiority over existing Rl and supervised methods [15,41,53]. 

cOMPUteR ASSiSteD SURGeRy 15 

**Vessel Tree Reconstruction:** Su et al. [33] combined DRl, cNNs, and vessel construction to extract anterior vessel trees from ctA/ctP images, achieving recall and precision around 68–70%. 

**Image Synthesis and Segmentation:** Xu et al. [30] proposed Pix-GRl to synthesize Gd-enhanced tumors on non-enhanced MRis, achieving 87.60% accuracy in tumor classification. Zhang et al. [26] developed wStS, a dual-strategy DRl framework with teacher-student learning, reaching Dice and Recall scores of 83.11% and 85.12%. 

**Visual Question Answering (VQA):** Fountoukidou et al. [32] introduced a beta-distribution-driven Rl interrogator for diabetic macular edema grading. it outperformed conventional vQA strategies and showed potential for broader clinical use. 

**Image Quality Assessment (IQA):** Saeed et al. [31] developed a meta-Rl framework that adapts iQA systems to specific downstream task requirements and new datasets. 

**Multimodal Image Registration:** Hu et al. [29] framed ct-MR registration as an Rl task, integrating convolutional lStM, similarity learning, and custom rewards for accurate alignment. 

**Cardiovascular Health Assessment:** lian et al. [28] introduced APRil, an Rl system with anatomical priors, achieving lower MAes for cAlD and ciMt compared to traditional techniques. 

**Ophthalmology Applications:** Kucur et al. [44] proposed PASS, an Rl-based perimetry system that reduces test locations and time by 30–70%, improving visual field reconstruction accuracy. 

**Breast Cancer Diagnosis:** Huang et al. [27] presented Aw3M, a Rl framework using multi-modal ultrasound and adaptive weighting to enhance diagnosis accuracy. 

**Fetal Imaging:** Fiorentino et al. [54] reviewed the application of cNNs, RNNs, and DRl in fetal ultrasound, highlighting their role in detecting standard planes. 

**Spine Imaging:** Zhang et al. [36] developed ScRl, a DRl method for sequential identification and segmentation of vertebral bodies, capturing spinal structure relationships. table 1 summarizes the findings from extracted papers (excluding 2 review papers). 

#### **_4.2. RL algorithms and policy Optimization in medical imaging (RQ2)_** 

this section summarizes key Rl algorithms used in RlMi (see Figures 10 and 12): 

#### **_4.2.1. RL-based algorithms_** 

Algorithms for reinforcement learning (Rl) fall into two categories: tabular and approximate approaches. Monte carlo, temporal difference, and dynamic programming are popular tabular techniques. 



<!-- Start of picture text -->
P Go to step 2, update the<br>Environment State nee mate state by new estimate,<br>(model free) Bellman Equn contiue untill the optimal<br>qun. value obtained<br><!-- End of picture text -->

**Figure 9.** Q learning algorithm. 



<!-- Start of picture text -->
RL based Algorithm<br>Tabular Methods<br>Dynamic Monts Parke Temporal<br>Programming Difference<br>Model, value Function Model Free<br>Policy Iteration Value Iteration Sarsa (A)<br>Random ActionA computedValue of the and states evaluated are<br><!-- End of picture text -->

**Figure 10.** rL based algorithms. 

16 M. B. SAMPA et Al. 



<!-- Start of picture text -->
Combination of Model<br>based and Model free RL<br>Algorithms (MCTS)<br>Approximate Methods<br><!-- End of picture text -->

**Figure 11.** Combination of model based and model free rL algorithms. 



<!-- Start of picture text -->
[um<br>[mw _| [_on_|<br>[oto [swore<br>[tig ||_S%*_]| ner [tine ||_mes [renrone| [ED<br>[tewten|["&°] [2026 [oem] [Sc [oors] [re]<br><!-- End of picture text -->

**Figure 12.** rL Algorithms used in medical image analysis. 

**_Tabular approaches_** : when using tabular approaches, the agent keeps track of values for states, actions, and state-action pairs in a table (or matrix). every entry in the table represents the estimated value of a particular state or pair of states and actions. these techniques are conceptually simple and easy to comprehend, but because the size of the table increases exponentially with the number of dimensions, they become unworkable when the state or action space is huge. Here is a description of common tabular techniques (Figure 10): 

_Q-learning_ : An off-policy, model-free method based on the Bellman equation [55]: 



where α is the learning rate. the algorithm of Q learning is shown in Figure 9. 

when dealing with discrete action spaces problems where the agent has a limited number of actions to pick from in each state Q-learning works very well. if the action space is continuous, modifications like Deep Q Networks (DQN) are often used. Q-learning is particularly useful in scenarios with unknown dynamics, sparse rewards, and evolving conditions (Figure 10). 

_Sarsa_ : An on-policy variant [55]: 



_Sarsa_ (𝜆): An extension of the Sarsa algorithm that incorporates eligibility traces to manage credit assignment over several time steps is called Sarsa (λ). the algorithm’s goal is to acquire policies for Markov Decision Processes (MDPs), in which an agent engages with its surroundings, acts, and is rewarded. eligibility traces are used by Sarsa (𝜆) to combine Monte carlo and temporal difference approaches. 

_R-learning_ : R-learning is an extension of Q-learning for continuous tasks using average reward [55–57]: 



where 𝜌 is the average reward and can be achieved by 

ρ ← ρ + [ _r_ − ρ + max _Q_ ( _s_ , _a_ ′)− max _Q_ ( _s_ , _a_ )] (4) 

where 𝛽 is the learning rate. 

cOMPUteR ASSiSteD SURGeRy 17 

#### **Monte Carlo tree search (MCTS):** 

combines model-free (Monte carlo or temporal difference) and model-based (Dynamic Programming) Rl _via_ iterative tree search [55,58,59]. 

**_Approximate approaches_** : Approximate approaches utilize function approximations to represent value functions or policies, such as neural networks. instead of storing values for every state or action, these methods generalize learning across a broader set of situations (Figure 11). 

_Fitted Q_ : Fitted Q (FQ) is an approximate reinforcement learning algorithm inspired by the concept of fitted value iteration suggested by [60]. it is a type of model-based approach that leverages function approximation to represent and estimate the action-value function in a Markov Decision Process (MDP). Fitted Q employs function approximation to estimate the action-value function (Q-function) for different states and actions in the MDP. instead of maintaining a table of Q-values for all possible state-action pairs, a function approximator is employed to generalize the Q-values across the state and action space. it allows the use of various regression techniques, and researchers can experiment with different algorithms based on the characteristics of their problem. For instance [61], employed randomized trees as a regression algorithm for function approximation in their variant of Fitted Q, which they referred to as fitted Q iteration (FQi). 

#### **_4.2.2. Reinforcement learning based medical imaging algorithm_** 

For policy optimization, Rl algorithms are employed. the following figure summarizes all algorithms of reinforcement learning. typically, we separate the algorithms utilized by RlMis into two categories: DRl-based and Rl-based. Many algorithms have been proposed to solve an Rl problem; these can be broadly divided into tabular and approximation approaches [7]. value functions can be expressed as tables since tabular approaches have smaller action and state spaces. However, because state space is so large, the aim of approximate approaches is to discover a good approximation while utilizing a minimal amount of processing power. it is important to highlight that DRl algorithms use deep learning for function approximation, which sets them apart from conventional Rl methods (a more detailed explanation on this is presented shortly in DRl-based Algorithms section). Here, we provide a quick overview of the Rl algorithms used by RlMis. these algorithms are shown in Figure 12. 

#### **_4.2.3. Survey of algorithms with respect to the RLMI framework_** 

Deep Q-Network is applied for landmark detection on medical images [62] because of static environments (i.e. fixed number of landmarks of training images) and discrete variables of action design. Deep Q-Network (DQN) with single agent (DeepSaQ) and multiple agent (DeepMaQ) have been proven to be effective in incomplete images scenario [25]. the PPO approach with the advantage actor-critic (A2c) framework is utilized to acquire the best policy in the vessel tracking application [33]. through the use of reinforcement learning to the process of selecting visual field (vF) locations, a policy function that employs the ReiNFORce rule is learned and improved to select locations iteratively that effectively reconstruct vFs at the end of a specified horizon [44]. A study by [24] evaluated various reinforcement learning algorithms with various training strategies for identifying anatomical landmarks in 3D images. the Rl agent learns to identify the optimal path to the landmark by interacting with an environment. Reinforcement learning algorithms like DQN, DDQN (Double DQN), Duel DQN, and Duel DDQN are among the particular training methods. it is discovered that the ideal DQN algorithm for attaining peak performance is contingent upon the surroundings. the study [40] compares four Rl algorithms: actor-critic policy gradient (A2c), deep deterministic policy gradient (DDPG), twin delayed deep deterministic policy gradient (tD3), and deep Q-network (DeepQN). A2c performed the best in both landmark detection and anatomical angle estimation. the ReiNFORce algorithm for policy optimization was used in the study [47] to detect breast lesions from ultrasound videos. twin-Delayed Deep-Deterministic Policy Gradient (tD3) and Soft Actor-critic (SAc) were utilized by [43] to optimize policy for analyzing the white-matter structural connectivity of a brain. the study [41] uses the deep Q-network (DQN) for real-time 3D landmark detection [42]. apply the ReiNFORce rule to recognize anatomical landmarks. instead of relying on DRl, they employed Rl to set the target precision using manually constructed criteria. the DeepQNetwork 

18 M. B. SAMPA et Al. 

technique was utilized in the paper [63] for object localization. the author of the paper [36] segmented and automatically identified vertebral structures from MRi images using the Soft Actor-critic (SAc) method. Monte carlo (Mc) learning and Q-learning were both employed by [32] to evaluate an automatic adaptive questioning approach. the study [31] evaluates image quality using meta-Rl. the A3c model was utilized in the study [29] for multimodal medical picture registration. DQN was used in the study [28] to estimate the cAlD & ciMt in order to diagnose cardiovascular disease risk. ReiNFORce rule was applied by [27] to diagnose breast cancer. 

### **5. Discussion** 

this systematic review synthesized evidence on the application of reinforcement learning in medical image analysis (RlMi). the following discussion provides a deeper interpretation and comparison of the findings, situates them within the broader literature, and highlights their implications for future research and applications. 

#### **_5.1. Performance comparison of different RL algorithms used in medical image analysis_** 

value-based methods like Q-learning are often simpler but may suffer from instability in high-dimensional state spaces, while policy gradient methods, such as Proximal Policy Optimization (PPO), are more stable but computationally expensive [64]. Some Rl algorithms are better suited for specific applications—e.g. DQN performs well in discrete action spaces [9], while PPO and tRPO are more suitable for continuous control tasks [65]. Providing a cross-study comparison that categorizes algorithms based on their success in different domains (e.g. healthcare, robotics, gaming) can further enhance the critical nature of the analysis. A study compares the Rl-based method with conventional approaches, showing that the proposed method outperforms other landmark detection models that do not incorporate Rl or shape priors, especially in scenarios where images are incomplete or noisy. Specifically, it achieved higher precision and recall in landmark localization. the method demonstrated robust precision when handling up to 80% of missing image data in dual-energy X-ray absorptiometry (DXA) scans, achieving an average distance error of 2.29 cm. in other cases, such as cardiac MRi and head ct datasets, the Rl model also exhibited strong detection capabilities even with unseen landmarks in regions outside the field of view, with an average distance error of 6.84 mm in the 3D half-head ct scans [25]. 

#### **_5.2. Proposed RLMI framework_** 

After carefully studying all collected reinforcement learning medical imaging (RlMis) articles, we identified four common components in each. An effective Reinforcement learning Medical imaging (RlMi) study should meticulously design and address these components. to standardize the development process of RlMis, we propose a framework comprising four key components: (1) State Representation, (2) Policy Optimization, (3) Reward Formulation, and (4) environment Building. each of these components is elaborated upon in the previous Section 2. 

#### **_5.2.1. State representation_** 

the agent’s state contains image information from previous steps and the corresponding displacement vector array at the adjacent position in the image. to generate an action at, the policy receives the state as input [43]. it is obtained from feature exaction, cropping small part of an image, predicted values etc. For instance, a sequence of clipped patches from an image is used by [25] to represent the states of the agents. the combined utilization of all clipped patches stabilizes the search trajectories. A DNN-based reinforcement learning model (like DeepMaQ) receives each patch (cropped slice) for each related agent to determine which action each agent should perform. 

three 21 × 21 x 21 sub-volumes comprise the state’s image data, and they are centered at the agent position at _pt_ and two prior agent positions, _pt_ −1 and _pt_ −2. this information is crucial for navigating 

cOMPUteR ASSiSteD SURGeRy 19 

through the volume, especially around large vessels and bifurcations. the state also incorporates a binary agent path and a displacement vector array, enabling directed tracking of vessels. From the current position to the initial position, the dynamic displacement vector array provides additional directional information. the 26 connected nearby voxels form the agent’s discrete action space, represented by scaled displacement vectors, ensuring that each agent action results in a step of at least two [33]. each state is defined as a 3D Region of interest (ROi) centered on the target landmark of a 3D image in [24] for the purpose of landmark detection. A frame history buffer is used to capture the last 4 action steps (ROis) taken by the agent in its search for the land- mark. this stabilizes the search trajectories and prevents the agent from getting stuck in repeated cycles. the last nine sliced planes that three agents anticipated are referred to as the states in [66]. in [40] the states are defined as image patch around the targeted landmark [47]. denoted states by key frame video, with a size of H × w ×c × N<sup>0</sup> , represents height, width, color channel respectively from breast ultra sound (US) video. the state is extracted from the diffusion signal in [43]. 

States S are defined by [67] as the collection of weights corresponding to different supervision layers. A softmax function is used to anticipate these weights, denoted by {αk}, by taking four real values for {βk}. ακ = ex4 <u>p(βκ</u> ) . ∑ k=1exp(βκ ) 

Xu et al. [30] defines states as the pixel of the image. According to [32] the state represents the history of the questions, the QS (interrogator) asked and the answers the Mue (responder) provided for a particular image. in [45] states of the capsule robot are defined as position and rotation vectors. 

#### **_5.2.2. Policy Optimization_** 

Policy Optimization in reinforcement learning (Rl) refers to a class of methods that aim to directly improve the agent’s policy that is, the strategy it uses to select actions in different states by optimizing the parameters of the policy to maximize the expected cumulative reward. in [25] DeepMaQ-Net is utilized to learn the navigation policy for every agent to find its target from a random starting position. in the case of vessel tracking training task [33] to learn the agent to tracking the vessel centerline the optimal policy is obtained using the PPO method with the advantage actor critic (A2c) framework. the study [44] used the policy-gradient reinforcement learning approach for the task of visual field re- construction from sparse observations. 

#### **_5.2.3. Environment Building_** 

environment is where everything is located. For example, the full image. Generally, we take a single or group of multimodality medical images input as the environment. For example, the environment in [33] is the normalized 3D ctA image. in [24] 3D image is used to represent the environment [66] defines environment as the 3D US volume. 

#### **_5.2.4. Agent_** 

State is the input of the agent. when a landmark is detected, an agent navigates to the target position by performing a series of actions. the action of agent j from the action space is designated as a j ∈ {moveleft, moveright, moveup, movedown}. this is because in [25] they consider four types of actions for 2D images, including move left, move right, move up, and move down with a preset step length. when the agent terminates at a specific scale, the moving length will decrease. An agent receives feedback from the environment with an action, and the feedback is implemented _via_ the reward mechanism. in [33] the agent’s discrete action space is defined as the 26 connected nearby voxels of the current position pt in an agent path. the agent follow the reference path Gt, which is a voxelized representation of the annotated vessel track. According to [24], the lower Q-value is used to determine the terminating state. the Q-values are found to be larger when the agent is distant from the target location and lower when the agent is closer. Higher Q-values are intuitively assigned by DQN to states that are far from the target landmark, and vice versa, encouraging the agent to act from those states. 

20 M. B. SAMPA et Al. 

#### **_5.2.5. Reward Formulation_** 

the reward function is an essential component of Rl to achieve the optimal policy. By interacting with an environment in a sequential manner, an agent in Rl is tasked with maximizing a numerical reward. An agent receives a reward for their efforts at current state, and the rewards vary depending on the actions taken by the agent at a different state. the reward function of multi- agent Rl is defined as a ɺ d t−1 d t concatenation of individual rewards from each agent, i.e. _R_ = {rj | j ∈ J}, where rj= D( x j , x j )<sup>−</sup> D( x j , x j ). x j d, is the ground-true location for agent j, x j t−1 and x j t are the locations of agent j before and after taking an action, respectively, and _D_ is euclidean distance. Keep in mind that if the unit is a pixel and the distance between the location and ground truth decreases, rj is positive [25]. According to [33], the reward of an action is determined by calculating the difference in the curve-to-curve distance between the agent path and the reference standard at time _t_ and time _t_ − 1. the reward function is defined as _R_ = _D_ ( _pi_ −1, _pt_ ) − _D_ ( _pi_ , _pt_ ), where _D_ is the euclidean distance between two points, _pt_ is the target ground truth landmark’s location, and _pi_ is the current anticipated landmark’s position at step _i_ [24]. the agent’s proximity to the desired target location can be determined by comparing the two euclidean distances, the one from the previous step and the one from the current step. According to yang et al. [66], the Reward for every agent, denoted as Rk ∈ {−1, 0, +1}, was computed as: Rk = sgn (D<sup>t</sup> −1k − Dt k<sup>), where sgn is</sup> a sign function and _t_ represents the iteration step in agent interacting ,D<sup>t</sup> k<sup>represents the difference</sup> between the plane parameters of plane p<sup>t</sup> k<sup>and that of its corresponding ground truthpg</sup> k<sup>. in order to</sup> terminate the agent-environment interaction and handle the last step prediction as the final output, the terminal states select a fixed-steps approach. 

#### **_5.3. Representative applications of RL in medical image analysis_** 

tumor Detection and Segmentations: A deep Q-learning model was implemented for brain tumor analysis where the agent interacted with segmented images and learned from its decisions. the model outperformed traditional methods, particularly for tumor localization and early-stage segmentation tasks, even under limited training data conditions (e.g., achieving approximately 70% accuracy versus 11% for a supervised keypoint-based detector) [68]. 

landmark Detection: in a study focused on detecting anatomical landmarks in cardiac MRi scans, an Rl-based agent was developed to locate specific points such as valve positions. the agent received rewards for correctly identifying landmarks and penalties for inaccuracies. this approach significantly improved landmark detection accuracy compared to traditional methods. the results indicated that multi-scale agents achieved lower localization errors compared to fixed-scale agents and outperformed traditional methods like stratified decision forests. For instance, the Duel DQN agent with a multi-scale strategy achieved a mean localization error of 4.42 ± 2.67 mm for the apex and 5.76 ± 3.89 mm for the mitral valve center. in contrast, the stratified decision forests method reported errors of 6.74 ± 4.12 mm for the apex and 6.32 ± 3.95 mm for the mitral valve center [24]. the Rl agent’s adaptive learning capabilities allowed it to generalize better across diverse patient datasets. 

Disease Diagnosis: the study reports that the proposed model achieved high accuracy rates in classifying the five stages of diabetic retinopathy, outperforming conventional methods. this suggests that the reinforcement learning-based approach provided a significant improvement over traditional techniques [69]. the adaptive nature of Rl allowed it to improve its diagnostic criteria over time, leading to more reliable assessments. 

image Segmentation: in the study [36], the reinforcement learning-based ScRl framework achieved a Dice coefficient of 92.6% for vertebral body segmentation. this result demonstrates that incorporating sequential anatomical modeling through Rl significantly enhances segmentation accuracy compared to traditional methods. 

#### **_5.4. Key challenges in medical image analysis_** 

Data Annotation and Scarcity: Annotating medical images is time consuming and often requires expert knowledge. Rl techniques, particularly semi-supervised and unsupervised methods, can reduce dependence on large, fully annotated datasets by leveraging unlabeled data for model training. 

cOMPUteR ASSiSteD SURGeRy 21 

complexity of Medical images: Medical images often involve complex, high-dimensional data. Rl’s ability to handle sequential decision-making and learn optimal strategies over time can improve tasks like object detection, landmark localization, and image segmentation, where navigating complex structures is crucial. 

Generalization and Adaptability: Many machine learning models struggle with generalization across different medical datasets and imaging modalities. Rl’s flexible learning strategies allow models to adapt to diverse datasets and tasks, improving performance across different clinical scenarios. 

Real-time Decision Making: in tasks like disease diagnosis and treatment planning, real-time decisions are essential. Rl, with its focus on optimizing long-term rewards, can be used to develop systems that assist clinicians in making quick, evidence-based decisions. 

#### **_5.5. Limitations of the current review study_** 

Despite its comprehensive scope, this review has several limitations. First, the coverage is limited to studies published up to September 2023, so recent developments in reinforcement learning for medical imaging may not be captured. Second, the search was confined to selected databases and english-language publications, introducing potential database and language biases that may have excluded relevant research. third, the included studies exhibited substantial heterogeneity in tasks, datasets, and evaluation metrics, which precluded quantitative synthesis (e.g. meta-analysis) and limited the review to qualitative assessment. At the study level, important limitations in the current body of research were identified. Many studies relied on small or single-center datasets, restricting generalizability. Reproducibility was limited, as few studies made code or data publicly available. computational considerations, including training time and hardware requirements, were rarely reported, despite being crucial for real-world feasibility. while some studies addressed clinical relevance, large-scale validation and integration into routine clinical practice remain limited. Methodologically, the review was restricted to publications from January 2019 to September 2023, and the formal search was conducted only in ieee Xplore, ScienceDirect, and Springerlink. Although informal checks were performed in Google Scholar, PubMed, and arXiv, some relevant studies may have been missed. Finally, while the review proposes a Reinforcement learning for Medical imaging (RlMi) framework and provides conceptual comparisons with existing studies, these remain interpretive rather than empirically validated, and future work is needed to test the framework in practical settings. 

#### **_5.6. Future directions (RQ3: What is the future direction?)_** 

Building upon the identified challenges and research gaps, several promising directions can be pursued to advance the integration of reinforcement learning (Rl) in medical image analysis: 

1. Algorithmic enhancements 

   - Future work should prioritize the development of more sample-efficient Rl approaches, such as model-based Rl and transfer learning, to reduce the dependency on large training datasets. Additionally, the integration of explainability and safety constraints into Rl algorithms will be crucial for ensuring clinical reliability and regulatory acceptance. 

2. Dataset expansion and Diversity 

the creation of large-scale, diverse, and well-annotated medical imaging datasets remains an urgent need. Alongside this, the adoption of synthetic data generation techniques can help address class imbalance and scarcity, particularly in rare diseases or underrepresented imaging modalities. 

3. Broadening Application Domains while Rl has demonstrated effectiveness in landmark detection, segmentation, and registration, future studies should extend its application to multimodal disease diagnosis, adaptive radiotherapy, and personalized treatment planning. Special emphasis should also be placed on leveraging Rl in low-resource settings and rare disease contexts, where traditional data-driven approaches often underperform. 

4. Dynamic and Multimodal Analysis 

   - Another key research avenue involves extending Rl to account for temporal dynamics in longitudinal imaging studies, thereby enabling more robust disease monitoring. Moreover, the 

22 M. B. SAMPA et Al. 

incorporation of multimodal information—including imaging data, genomic profiles, and clinical records—could facilitate more holistic and patient-centered decision-making. 

### **6. Conclusion** 

this systematic review highlights the transformative potential of reinforcement learning (Rl) and deep reinforcement learning (DRl) in medical image analysis, demonstrating their ability to tackle complex, sequential decision-making tasks across a wide range of imaging modalities and clinical applications. evidence from 20 peer-reviewed studies between 2019 and 2023 shows that Rl-based approaches— including Deep Q-Networks (DQN) and its variants, Proximal Policy Optimization (PPO), Advantage Actor-critic (A2c), ReiNFORce, twin-Delayed Deep Deterministic Policy Gradient (tD3), and Soft Actor-critic (SAc)—have achieved remarkable performance in landmark detection, image segmentation, lesion identification, disease diagnosis, image registration, and cardiovascular evaluation. Multi-agent Rl systems and Rl-augmented convolutional neural networks (cNNs) have particularly demonstrated robustness, accuracy, and real-time capabilities across ultrasound, MRi, ct, and X-ray imaging, often outperforming conventional machine learning methods while reducing reliance on large annotated datasets. 

the findings underscore the importance of a unified Rl framework for medical imaging (RlMi), comprising state representation, policy optimization, reward formulation, and environment modeling, to ensure stable policy learning, efficient navigation, and task adaptability across clinical contexts. Rl also shows promise in enabling multimodal image registration, 3D landmark detection, target precision tuning, and risk estimation, highlighting its broad clinical applicability. 

Despite these advances, several challenges remain, including improving model interpretability, enhancing generalizability across diverse patient populations, optimizing task-specific algorithms, incorporating temporal and anatomical context, and addressing data scarcity in real-world clinical settings. 

looking forward, the integration of Rl with meta-learning, hybrid architectures, and human-in-the-loop systems offers promising avenues for creating intelligent, adaptive, and clinically meaningful medical imaging solutions. the continued collaboration of clinicians, computer scientists, and data scientists will be essential to translating Rl from experimental research to routine clinical practice. 

Rl and DRl are poised to revolutionize medical image analysis, offering enhanced accuracy, adaptability, and efficiency. while the field is still evolving, these approaches hold substantial potential to advance precision diagnosis, personalized healthcare, and autonomous medical imaging systems, paving the way for the next generation of Ai-driven clinical decision support. 

### **Author contributions** 

Masuda Begum Sampa: conceptualization, methodology, data curation, writing—original draft preparation, writing— review and editing; Nor Hidayati Abdul Aziz: Study selection, data extraction, supervision, funding acquisition; Siddikur Rahman: writing—review and editing; Azlina Ab. Aziz: Study selection, data extraction, supervision, funding acquisition; Rosli Besar: writing—review and editing; Anith Khairunnisa Ghazali: writing—review and editing. All the authors have read and agreed to the published version of the manuscript. All authors approved the final version to be published. 

### **Disclosure statement** 

No potential conflict of interest was reported by the author(s). 

### **Funding** 

this study was funded by Multimedia University, Malaysia (grant number MMUi/240090). 

### **References** 

- [1] wen J, Zhang Z, lan y, et al. A survey on federated learning: challenges and applications. int J Mach learn cybern. 2023;14(2):513–535. doi:10.1007/s13042-022-01647-y. 

- [2] Jiang Z, Salcudean Se, Navab N. Robotic ultrasound imaging: state-of-the-art and future perspectives. Med image Anal. 2023;89:102878. doi:10.1016/j.media.2023.102878. 

cOMPUteR ASSiSteD SURGeRy 23 

- [3] Jebamikyous H, li M, Suhas y, et al. leveraging machine learning and blockchain in e-commerce and beyond: benefits, models, and application. Discov Artif intell. 2023;3(3):1–16. doi:10.1007/s44163-022-00046-0. 

- [4] Dalla Pozza N, Buffoni l, Martina S, et al. Quantum reinforcement learning: the maze problem. Quantum Mach intell. 2022;4(11):1–10. doi:10.1007/s42484-022-00068-y. 

- [5] Mall PK, Singh PK. MtR-SDl: a soft computing based multi-tier rank model for shoulder X-ray classification. Soft comput. 2024;28(Suppl 2):621. doi:10.1007/s00500-023-08562-6. 

- [6] Sutton RS, Barto AG. Reinforcement learning: an introduction. cambridge (MA): Mit Press; 2015. 

- [7] Afsar MM, crump t, Far B. Reinforcement learning based recommender systems: a survey. AcM comput Surv. 2023;55(7):1–38. doi:10.1145/3543846. 

- [8] Kaelbling lP, littman Ml, Moore Aw. Reinforcement learning: a survey. jair. 1996;4:237–285. doi:10.1613/jair.301. 

- [9] Mnih v, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning. Nature. 2015;518(7540):529–533. doi:10.1038/nature14236. 

- [10] lillicrap tP, Hunt JJ, Pritzel A, et al. continuous control with deep reinforcement learning. in Proc int conf learn Representations; 2016. 

- [11] Silver D, Schrittwieser J, Simonyan K, et al. Mastering the game of Go without human knowledge. Nature. 2017;550(7676):354–359. doi:10.1038/nature24270. 

- [12] Sahba F, tizhoosh HR, Salama MMA. Application of reinforcement learning for segmentation of transrectal ultrasound images. BMc Med imaging. 2008;8(1):8. doi:10.1186/1471-2342-8-8. 

- [13] chitsaz M, woo cS. time spent for housework and child care by husbands with children under six years old. Popul Res Policy Rev. 2011;26(2):247–255. 

- [14] wang l, lekadir K, lee Sl, et al. A general framework for context-specific image segmentation using reinforcement learning. ieee trans Med imaging. 2013;32(5):943–956. doi:10.1109/tMi.2013.2252431. 

- [15] Ghesu Fc, Georgescu B, Mansi t, et al. An artificial agent for anatomical landmark detection in medical images. in: Proc Med image comput comput Assist interv; 2016. p. 229–237. 

- [16] Neumann D, Mansi t, itu l, et al. Medical image computing and computer-assisted intervention—MiccAi 2015. lect Notes comput Sci. 2015;9350:442–449. doi:10.1007/978-3-319-24571-3. 

- [17] Neumann D, Mansi t, itu l, et al. A self-taught artificial agent for multi-physics computational model personalization. Med image Anal. 2016;34:52–64. doi:10.1016/j.media.2016.04.003. 

- [18] chen X, yao l, McAuley J, et al. A survey of deep reinforcement learning in recommender systems: a systematic review and future directions. J Acm. 2021;37(4):111. 

- [19] Musa N, Gital Ay, Aljojo N, et al. Applications of deep learning in electrocardiogram analysis: a systematic review. J Ambient intell Humaniz comput. 2023;14(7):9677–9750. doi:10.1007/s12652-022-03868-z. 

- [20] valdez F, Melin P. A review on quantum computing and deep learning algorithms and their applications. Soft comput. 2023;27:13217–13236. doi:10.1007/s00500-022-07037-4. 

- [21] Zhou SK, le HN, luu K, et al. Deep reinforcement learning in medical imaging: a literature review. Med image Anal. 2021;73:102193. doi:10.1016/j.media.2021.102193. 

- [22] Srinidhi cl, ciga O, Martel Al. Deep neural network models for computational histopathology: a survey. Med image Anal. 2021;67:101813. doi:10.1016/j.media.2020.101813. 

- [23] Hu M, Zhang J, Matkovic l, et al. Reinforcement learning in medical image analysis: concepts, applications, challenges, and future directions. J Appl clin Med Phys. 2023;24(2):e13898. doi:10.1002/acm2.13898. 

- [24] Alansary A, Oktay O, li y, et al. evaluating reinforcement learning agents for anatomical landmark detection. Med image Anal. 2019;53:156–164. doi:10.1016/j.media.2019.02.007. 

- [25] wan K, li l, Jia D, et al. Multi-target landmark detection with incomplete images via reinforcement learning and shape prior embedding. Med image Anal. 2023;89:102875. doi:10.1016/j.media.2023.102875. 

- [26] Zhang D, chen B, chong J, et al. weakly supervised teacher–student network for liver tumor segmentation from non-enhanced images. Med image Anal. 2021;70:102005. doi:10.1016/j.media.2021.102005. 

- [27] Huang R, lin Z, Dou H, et al. Aw3M: an auto-weighting and recovery framework for breast cancer diagnosis using multi-modal ultrasound. Med image Anal. 2021;72:102137. doi:10.1016/j.media.2021.102137. 

- [28] lian S, luo Z, Feng c, et al. APRil: anatomical prior-guided reinforcement learning for accurate carotid lumen diameter and intima-media thickness measurement. Med image Anal. 2021;71:102040. doi:10.1016/j.media.2021.102040. 

- [29] Hu J, luo Z, wang X, et al. end-to-end multimodal image registration via reinforcement learning. Med image Anal. 2021;68:101878. doi:10.1016/j.media.2020.101878. 

- [30] Xu c, Zhang D, chong J, et al. Synthesis of gadolinium-enhanced liver tumors on nonenhanced liver MR images using pixel-level graph reinforcement learning. Med image Anal. 2021;69:101976. doi:10.1016/j.media.2021.101976. 

- [31] Saeed SU, Fu y, Stavrinides v, et al. image quality assessment for machine learning tasks using meta-reinforcement learning. Med image Anal. 2022;78:102427. doi:10.1016/j.media.2022.102427. 

- [32] Fountoukidou t, Sznitman R. A reinforcement learning approach for vQA validation: an application to diabetic macular edema grading. Med image Anal. 2023;87:102822. doi:10.1016/j.media.2023.102822. 

- [33] Su J, li S, wolff l, et al. Deep reinforcement learning for cerebral anterior vessel tree extraction from 3D ctA images. Med image Anal. 2023;84:102724. doi:10.1016/j.media.2022.102724. 

- [34] Macaulay MO, Shafiee M. Machine learning techniques for robotic and autonomous inspection of mechanical systems and civil infrastructure. Auton intell Syst. 2022;2(8):1–25. doi:10.1007/s43684-022-00025-3. 

- [35] Arulkumaran K, Deisenroth MP, Brundage M, et al. A brief survey of deep reinforcement learning. ieee Signal Process Mag. 2017;34(6):26–38. doi:10.1109/MSP.2017.274324. 

24 M. B. SAMPA et Al. 

- [36] Zhang D, chen B, li S. Sequential conditional reinforcement learning for simultaneous vertebral body detection and segmentation with modeling the spine anatomy. Med image Anal. 2021;67:101861. doi:10.1016/j.media.2020.101861. 

- [37] Kitchenham B, charters S. Guidelines for performing systematic literature reviews in software engineering. Keele (UK): Keele University; 2007:45. 

- [38] Brereton P, Kitchenham BA, Budgen D, et al. lessons from applying the systematic literature review process within the software engineering domain. J Syst Softw. 2007;80(4):571–583. doi:10.1016/j.jss.2006.07.009. 

- [39] Xiao y, watson M. Guidance on conducting a systematic literature review. J Plan educ Res. 2019;39(1):93–112. doi:10.1177/0739456X17723971. 

- [40] Bekkouch iei, Maksudov B, Kiselev S, et al. Multi-landmark environment analysis with reinforcement learning for pelvic abnormality detection and quantification. Med image Anal. 2022;78:102417. doi:10.1016/j.media.2022.102417. 

- [41] Ghesu F-c, Georgescu B, Zheng y, et al. Multi-scale deep reinforcement learning for real-time 3D landmark detection in ct scans. ieee trans Pattern Anal Mach intell. 2019;41(1):176–189. doi:10.1109/tPAMi.2017.2782687. 

- [42] Zhou G-Q, Miao J, yang X, et al. learn fine-grained adaptive loss for multiple anatomical landmark detection in medical images. ieee J Biomed Health inform. 2021;25(10):3854–3864. doi:10.1109/JBHi.2021.3080703. 

- [43] théberge A, Desrosiers c, Descoteaux M, et al. track-to-learn: a general framework for tractography with deep reinforcement learning. Med image Anal. 2021;72:102093. doi:10.1016/j.media.2021.102093. 

- [44] Kucur ŞS, Márquez-Neila P, Abegg M, et al. Patient-attentive sequential strategy for perimetry-based visual field acquisition. Med image Anal. 2019;54:179–192. doi:10.1016/j.media.2019.03.002. 

- [45] İncetan K, celik iO, Obeid A, et al. vR-caps: a virtual environment for capsule endoscopy. Med image Anal. 2021;70:101990. doi:10.1016/j.media.2021.101990. 

- [46] wright R, Gomez A, Zimmer vA, et al. Fast fetal head compounding from multi-view 3D ultrasound. Med image Anal. 2023;89:102793. doi:10.1016/j.media.2023.102793. 

- [47] Huang R, ying Q, lin Z, et al. extracting keyframes of breast ultrasound video using deep reinforcement learning. Med image Anal. 2022;80:102490. doi:10.1016/j.media.2022.102490. 

- [48] Alansary A, le Folgoc l, vaillant G, et al. Automatic view planning with multi-scale deep reinforcement learning agents. Med image Anal. 2018;46:1–15. 

- [49] Maicas G, carneiro G, Bradley AP, et al. Deep reinforcement learning for active breast lesion detection from Dce-MRi. in: Proc Med image comput comput Assist interv; 2017. p. 665–673. 

- [50] liao R, Miao S, De tournemire P, et al. An artificial agent for robust image registration. AAAi. 2017;31(1):4168– 4175. doi:10.1609/aaai.v31i1.11230. 

- [51] Zhan y, Peng Z, Hermosillo G, et al. Robust multi-landmark detection based on information theoretic scheduling. in Medical image recognition, segmentation and parsing. the elsevier and MiccAi Society Book Series. london (UK): Academic Press; 2016. p. 45–70. 

- [52] Seghers D, Slagmolen P, lambelin y, et al. landmarkbased liver segmentation using local shape and intensity models. in: Proc MiccAi workshop; 2007. p. 135–142. 

- [53] li y, Alansary A, cerrolaza J, et al. Fast multiple landmark localisation using a patch-based iterative network. Med image Anal. 2018. 

- [54] Fiorentino Mc, villani FP, Di cosmo M, et al. A review on deep-learning algorithms for fetal ultrasound-image analysis. Med image Anal. 2023;83:102629. doi:10.1016/j.media.2022.102629. 

- [55] Sutton RS, Barto AG. Reinforcement learning: an introduction. 2nd ed. cambridge (MA): Mit Press; 2018. 

- [56] Schwartz A. A reinforcement learning method for maximizing undiscounted rewards. in Proceedings of the 10th international conference on Machine learning; 1993. p. 298–305. 

- [57] Moling O, Baltrunas l, Ricci F.6th Optimal radio channel recommendations with explicit and implicit feedback. in Proceedings of the AcM conference on Recommender Systems; 2012. p. 75–82. doi:10.1145/2365952.2365971. 

- [58] chaslot G, Bakkes S, Szita i, et al. Monte-carlo tree search: a new framework for game Ai. AiiDe. 2008;4(1):216– 217. doi:10.1609/aiide.v4i1.18700. 

- [59] Browne cB, Powley e, whitehouse D, et al. A survey of Monte carlo tree search methods. ieee trans comput intell Ai Games. 2012;4(1):1–43. doi:10.1109/tciAiG.2012.2186810. 

- [60] Gordon GJ. Approximate solutions to Markov decision processes. Pittsburgh (PA): carnegie Mellon University; 1999. technical Report cMU-cS-99-143. 

- [61] ernst D, Geurts P, wehenkel l. tree-based batch mode reinforcement learning. J Mach learn Res. 2005;6:503–556. 

- [62] yang H, Xie X. An actor–critic deep reinforcement learning approach for transmission scheduling in cognitive internet of things systems. ieee Syst J. 2020;14(1):51–60. doi:10.1109/JSySt.2019.2891520. 

- [63] caicedo Jc, lazebnik S. Active object localization with deep reinforcement learning. in Proceedings of the ieee international conference on computer vision; 2015. p. 2488–2496. doi:10.1109/iccv.2015.286. 

- [64] Schulman J, wolski F, Dhariwal P, et al. Proximal policy optimization algorithms. arXiv:1707.06347; 2017. 

- [65] Schulman J, levine S, Moritz P, et al. trust region policy optimization in Proceedings of the 32nd international conference on Machine learning; 2015. p. 1889–1897. 

- [66] yang X, Huang y, Huang R, et al. Searching collaborative agents for multi-plane localization in 3D ultrasound. Med image Anal. 2021;72:102119. doi:10.1016/j.media.2021.102119. 

- [67] wang K-N, yang X, Miao J, et al. AwSnet: an auto-weighted supervision attention network for myocardial scar and edema segmentation in multi-sequence cardiac magnetic resonance images. Med image Anal. 2022; 77:102362. doi:10.1016/j.media.2022.102362. 

cOMPUteR ASSiSteD SURGeRy 25 

- [68] Stember JN, Shalu H. Reinforcement learning using Deep Q networks and Q learning accurately localizes brain tumors on MRi with very small training sets. _BMC Medical Imaging_ . 2022;22:224. doi:10.1186/s12880-022-00919-x. 

- [69] tariq M, Palade v, Ma y, et al. Diabetic retinopathy detection using transfer and reinforcement learning with effective image preprocessing and data augmentation techniques. in Fusion of machine learning paradigms. cham: Springer; 2023. p. 33–61. doi:10.1007/978-3-031-22371-6_3. 

