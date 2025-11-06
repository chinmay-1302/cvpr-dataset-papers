<details>
<summary><a href="https://ieeexplore.ieee.org/document/9156412?denied=">nuScenes: A Multimodal Dataset for Autonomous Driving</a></summary>
<p>Robust detection and tracking of objects is crucial for the deployment of autonomous vehicle technology. Image based benchmark datasets have driven development in computer vision tasks such as object detection, tracking and segmentation of agents in the environment. Most autonomous vehicles, however, carry a combination of cameras and range sensors such as lidar and radar. As machine learning based methods for detection and tracking become more prevalent, there is a need to train and evaluate such methods on datasets containing range sensor data along with images. In this work we present nuTonomy scenes (nuScenes), the first dataset to carry the full autonomous vehicle sensor suite: 6 cameras, 5 radars and 1 lidar, all with full 360 degree field of view. nuScenes comprises 1000 scenes, each 20s long and fully annotated with 3D bounding boxes for 23 classes and 8 attributes. It has 7x as many annotations and 100x as many images as the pioneering KITTI dataset. We define novel 3D detection and tracking metrics. We also provide careful dataset analysis as well as baselines for lidar and image based detection and tracking. Data, development kit and more information are available online.</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/9156973?denied=">
Scalability in Perception for Autonomous Driving: Waymo Open Dataset
</a></summary>
<p>
The research community has increasing interest in autonomous driving research, despite the resource intensity of obtaining representative real world data. Existing self-driving datasets are limited in the scale and variation of the environments they capture, even though generalization within and between operating regions is crucial to the over-all viability of the technology. In an effort to help align the research community’s contributions with real-world self-driving problems, we introduce a new large scale, high quality, diverse dataset. Our new dataset consists of 1150 scenes that each span 20 seconds, consisting of well synchronized and calibrated high quality LiDAR and camera data captured across a range of urban and suburban geographies. It is 15x more diverse than the largest camera+LiDAR dataset available based on our proposed diversity metric. We exhaustively annotated this data with 2D (camera image) and 3D (LiDAR) bounding boxes, with consistent identifiers across frames. Finally, we provide strong baselines for 2D as well as 3D detection and tracking tasks. We further study the effects of dataset size and generalization across geographies on 3D detection methods. Find data, code and more up-to-date information at http://www.waymo.com/open.
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/9156329
">
BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning
</a></summary>
<p>
Datasets drive vision progress, yet existing driving datasets are impoverished in terms of visual content and supported tasks to study multitask learning for autonomous driving. Researchers are usually constrained to study a small set of problems on one dataset, while real-world computer vision applications require performing tasks of various complexities. We construct BDD100K, the largest driving video dataset with 100K videos and 10 tasks to evaluate the exciting progress of image recognition algorithms on autonomous driving. The dataset possesses geographic, environmental, and weather diversity, which is useful for training models that are less likely to be surprised by new conditions. Based on this diverse dataset, we build a benchmark for heterogeneous multitask learning and study how to solve the tasks together. Our experiments show that special training strategies are needed for existing models to perform such heterogeneous tasks. BDD100K opens the door for future studies in this important venue.
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/9879019
">
AutoMine: An Unmanned Mine Dataset
</a></summary>
<p>
Autonomous driving datasets have played an important role in validating the advancement of intelligent vehicle algorithms including localization, perception and prediction in academic areas. However, current existing datasets pay more attention to the structured urban road, which hampers the exploration on unstructured special scenarios. Moreover, the open-pit mine is one of the typical representatives for them. Therefore, we introduce the Autonomous driving dataset on the Mining scene (AutoMine) for positioning and perception tasks in this paper. The AutoMine is collected by multiple acquisition platforms including an SUV, a wide-body mining truck and an ordinary mining truck, depending on the actual mine operation scenarios. The dataset consists of 18+ driving hours, 18K annotated lidar and image frames for 3D perception with various mines, time-of-the-day and weather conditions. The main contributions of the AutoMine dataset are as follows: I.The first autonomous driving dataset for perception and localization in mine scenarios. 2.There are abundant dynamic obstacles of 9 degrees of freedom with large dimension difference (mining trucks and pedestrians) and extreme climatic conditions (the dust and snow) in the mining area. 3.Multi-platform acquisition strategies could capture mining data from multiple perspectives that fit the actual operation. More details can be found in our website(https://automine.cc).
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/9157053
">
Google Landmarks Dataset v2 – A Large-Scale Benchmark for Instance-Level Recognition and Retrieval
</a></summary>
<p>
While image retrieval and instance recognition techniques are progressing rapidly, there is a need for challenging datasets to accurately measure their performance -- while posing novel challenges that are relevant for practical applications. We introduce the Google Landmarks Dataset v2 (GLDv2), a new benchmark for large-scale, fine-grained instance recognition and image retrieval in the domain of human-made and natural landmarks. GLDv2 is the largest such dataset to date by a large margin, including over 5M images and 200k distinct instance labels. Its test set consists of 118k images with ground truth annotations for both the retrieval and recognition tasks. The ground truth construction involved over 800 hours of human annotator work. Our new dataset has several challenging properties inspired by real-world applications that previous datasets did not consider: An extremely long-tailed class distribution, a large fraction of out-of-domain test photos and large intra-class variability. The dataset is sourced from Wikimedia Commons, the world's largest crowdsourced collection of landmark photos. We provide baseline results for both recognition and retrieval tasks based on state-of-the-art methods as well as competitive results from a public challenge. We further demonstrate the suitability of the dataset for transfer learning by showing that image embeddings trained on it achieve competitive retrieval performance on independent datasets. The dataset images, ground-truth and metric scoring code are available at https://github.com/cvdfoundation/google-landmark
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/10205454
">
Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method
</a></summary>
<p>
Image aesthetics assessment (IAA) is a challenging task due to its highly subjective nature. Most of the current studies rely on large-scale datasets (e.g., AVA and AADB) to learn a general model for all kinds of photography images. However, little light has been shed on measuring the aesthetic quality of artistic images, and the existing datasets only contain relatively few artworks. Such a defect is a great obstacle to the aesthetic assessment of artistic images. To fill the gap in the field of artistic image aesthetics assessment (AIAA), we first introduce a large-scale AIAA dataset: Boldbrush Artistic Image Dataset (BAlD), which consists of 60,337 artistic images covering various art forms, with more than 360,000 votes from online users. We then propose a new method, SAAN (Style-specific Art Assessment Network), which can effectively extract and utilize style-specific and generic aesthetic information to evaluate artistic images. Experiments demonstrate that our proposed approach outperforms existing lAA methods on the proposed BAlD dataset according to quantitative comparisons. We believe the proposed dataset and method can serve as a foundation for future AIAA works and inspire more research in this field. Dataset and code are available at: https://github.com/Dreemurr-T/BAID.git
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/10656008
">
Towards Modern Image Manipulation Localization: A Large-Scale Dataset and Novel Methods
</a></summary>
<p>
In recent years, image manipulation localization has attracted increasing attention due to its pivotal role in guaranteeing social media security. However, how to accurately identify the forged regions remains an open challenge. One of the main bottlenecks lies in the severe scarcity of high-quality data, due to its costly creation process. To address this limitation, we propose a novel paradigm, termed as CAAA, to automatically and precisely annotate the numerous manually forged images from the web at the pixel level. We further propose a novel metric QES to facilitate the automatic filtering of unreliable annotations. With CAAA and QES, we construct a large-scale, diverse, and high-quality dataset comprising 123,150 manually forged images with mask annotations. Besides, we develop a new model APSC-Net for accurate image manipulation localization. According to extensive experiments, our dataset significantly improves the performance of various models on the widely-used benchmarks and such improvements are attributed to our proposed effective methods. The dataset and code are publicly available at https://github.com/qcf-568/MIML.
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/10204595
">
JRDB-Pose: A Large-Scale Dataset for Multi-Person Pose Estimation and Tracking
</a></summary>
<p>
Autonomous robotic systems operating in human environments must understand their surroundings to make accurate and safe decisions. In crowded human scenes with close-up human-robot interaction and robot navigation, a deep understanding of surrounding people requires reasoning about human motion and body dynamics over time with human body pose estimation and tracking. However, existing datasets captured from robot platforms either do not provide pose annotations or do not reflect the scene distribution of social robots. In this paper, we introduce JRDB-Pose, a large-scale dataset and benchmark for multi-person pose estimation and tracking. JRDB-Pose extends the existing JRDB which includes videos captured from a social navigation robot in a university campus environment, containing challenging scenes with crowded indoor and outdoor locations and a diverse range of scales and occlusion types. JRDB-Pose provides human pose annotations with per-keypoint occlusion labels and track IDs consistent across the scene and with existing annotations in JRDB. We conduct a thorough experimental study of state-of-the-art multi-person pose estimation and tracking methods on JRDB-Pose, showing that our dataset imposes new challenges for the existing methods. JRDB-Pose is available at https://jrdb.erc.monash.edu/.
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/9156686
">
DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection
</a></summary>
<p>
We present our on-going effort of constructing a large- scale benchmark for face forgery detection. The first version of this benchmark, DeeperForensics-1.0, represents the largest face forgery detection dataset by far, with 60, 000 videos constituted by a total of 17.6 million frames, 10 times larger than existing datasets of the same kind. Extensive real-world perturbations are applied to obtain a more challenging benchmark of larger scale and higher diversity. All source videos in DeeperForensics-1.0 are carefully collected, and fake videos are generated by a newly proposed end-to-end face swapping framework. The quality of generated videos outperforms those in existing datasets, validated by user studies. The benchmark features a hidden test set, which contains manipulated videos achieving high deceptive scores in human evaluations. We further contribute a comprehensive study that evaluates five representative detection baselines and make a thorough analysis of different settings.
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/5995586
">
A large-scale benchmark dataset for event recognition in surveillance video
</a></summary>
<p>
We introduce a new large-scale video dataset designed to assess the performance of diverse visual event recognition algorithms with a focus on continuous visual event recognition (CVER) in outdoor areas with wide coverage. Previous datasets for action recognition are unrealistic for real-world surveillance because they consist of short clips showing one action by one individual [15, 8]. Datasets have been developed for movies [11] and sports [12], but, these actions and scene conditions do not apply effectively to surveillance videos. Our dataset consists of many outdoor scenes with actions occurring naturally by non-actors in continuously captured videos of the real world. The dataset includes large numbers of instances for 23 event types distributed throughout 29 hours of video. This data is accompanied by detailed annotations which include both moving object tracks and event examples, which will provide solid basis for large-scale evaluation. Additionally, we propose different types of evaluation modes for visual recognition tasks and evaluation metrics along with our preliminary experimental results. We believe that this dataset will stimulate diverse aspects of computer vision research and help us to advance the CVER tasks in the years ahead.
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/9156308
">
AnimalWeb: A Large-Scale Hierarchical Dataset of Annotated Animal Faces
</a></summary>
<p>
Several studies show that animal needs are often expressed through their faces. Though remarkable progress has been made towards the automatic understanding of human faces, this has not been the case with animal faces. There exists significant room for algorithmic advances that could realize automatic systems for interpreting animal faces. Besides scientific value, resulting technology will foster better and cheaper animal care. We believe the underlying research progress is mainly obstructed by the lack of an adequately annotated dataset of animal faces, covering a wide spectrum of animal species. To this end, we introduce a large-scale, hierarchical annotated dataset of animal faces, featuring 22.4K faces from 350 diverse species and 21 animal orders across biological taxonomy. These faces are captured `in-the-wild' conditions and are consistently annotated with 9 landmarks on key facial features. The dataset is structured and scalable by design; its development underwent four systematic stages involving rigorous, overall effort of over 6K man-hours. We benchmark it for face alignment using the existing art under two new problem settings. Results showcase its challenging nature, unique attributes and present definite prospects for novel, adaptive, and generalized face-oriented CV algorithms. Further benchmarking the dataset across face detection and fine-grained recognition tasks demonstrates its multi-task applications and room for improvement. The dataset is available at: https://fdmaproject.wordpress.com/.
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/11094675
">
NSD-Imagery: A benchmark dataset for extending fMRI vision decoding methods to mental imagery
</a></summary>
<p>
We release NSD-Imagery, a benchmark dataset of human fMRI activity paired with mental images, to complement the existing Natural Scenes Dataset (NSD), a large-scale dataset of fMRI activity paired with seen images that enabled unprecedented improvements in fMRI-to-image reconstruction efforts. Recent models trained on NSD have been evaluated only on seen image reconstruction. Using NSD-Imagery, it is possible to assess how well these models perform on mental image reconstruction. This is a challenging generalization requirement because mental images are encoded in human brain activity with relatively lower signal-to-noise and spatial resolution; however, generalization from seen to mental imagery is critical for real-world applications in medical domains and brain-computer interfaces, where the desired information is always internally generated. We provide benchmarks for a suite of recent NSD-trained open-source visual decoding models (MindEye1, MindEye2, Brain Diffuser, iCNN, Takagi et al.) on NSD-Imagery, and show that the performance of decoding methods on mental images is largely decoupled from performance on vision reconstruction. We further demonstrate that architectural choices significantly impact cross-decoding performance: models employing simple linear decoding architectures and multi-modal feature decoding generalize better to mental imagery, while complex architectures tend to overfit visual training data. Our findings indicate that mental imagery datasets are critical for the development of practical applications, and establish NSD-Imagery as a useful resource for better aligning visual decoding methods with this goal.
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/10658200
">
Real-IAD: A Real-World Multi-View Dataset for Benchmarking Versatile Industrial Anomaly Detection
</a></summary>
<p>
Industrial anomaly detection (I AD) has garnered signif-icant attention and experienced rapid development. However, the recent development of I AD approach has encountered certain difficulties due to dataset limitations. On the one hand, most of the state-of-the-art methods have achieved saturation (over 99% in AUROC) on mainstream datasets such as MVTec, and the differences of methods cannot be well distinguished, leading to a significant gap between public datasets and actual application scenarios. On the other hand, the research on various new practical anomaly detection settings is limited by the scale of the dataset, posing a risk of overfitting in evaluation results. Therefore, we propose a large-scale, Real-world, and multi-view Industrial Anomaly Detection dataset, named Real- I AD, which contains 150K high-resolution images of 30 different objects, an order of magnitude larger than existing datasets. It has a larger range of defect area and ratio proportions, making it more challenging than previous datasets. To make the dataset closer to real application scenarios, we adopted a multi-view shooting method and proposed sample-level evaluation metrics. In addition, beyond the general unsupervised anomaly detection setting, we propose a new setting for Fully Unsupervised Indus-trial Anomaly Detection (FUIAD) based on the observation that the yield rate in industrial production is usually greater than 60%, which has more practical application value. Finally, we report the results of popular I AD methods on the Real- I AD dataset, providing a highly challenging benchmark to promote the development of the I AD field.
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/10658179
">
TUMTraf V2X Cooperative Perception Dataset
</a></summary>
<p>
Cooperative perception offers several benefits for en-hancing the capabilities of autonomous vehicles and im-proving road safety. Using roadside sensors in addition to onboard sensors increases reliability and extends the sensor range. External sensors offer higher situational awareness for automated vehicles and prevent occlusions. We propose CoopDet3D, a cooperative multi-modal fusion model, and TUMTraf- V2X, a perception dataset, for the cooperative 3D object detection and tracking task. Our dataset contains 2,000 labeled point clouds and 5,000 labeled images from five roadside and four onboard sensors. It includes 30k 3D boxes with track IDs and precise GPS and IMU data. We labeled nine categories and covered occlusion scenarios with challenging driving maneuvers, like traffic violations, near-miss events, overtaking, and U-turns. Through multiple experiments, we show that our CoopDet3D camera-LiDARfusion model achieves an increase of +14.36 3D mAP compared to a vehicle camera-LiDARfusion model. Finally, we make our dataset, model, labeling tool, and devkit publicly available on our website.
</p>
</details>
<details>
<summary>
<a href="
https://ieeexplore.ieee.org/document/9880410
">
GrainSpace: A Large-scale Dataset for Fine-grained and Domain-adaptive Recognition of Cereal Grains
</a></summary>
<p>
Cereal grains are a vital part of human diets and are important commodities for people's livelihood and international trade. Grain Appearance Inspection (GAI) serves as one of the crucial steps for the determination of grain quality and grain stratification for proper circulation, storage and food processing, etc. GAI is routinely performed manually by qualified inspectors with the aid of some hand tools. Automated GAI has the benefit of greatly assisting inspectors with their jobs but has been limited due to the lack of datasets and clear definitions of the tasks. In this paper we formulate GAI as three ubiquitous computer vision tasks: fine-grained recognition, domain adaptation and out-of-distribution recognition. We present a large-scale and publicly available cereal grains dataset called GrainSpace. Specifically, we construct three types of device prototypes for data acquisition, and a total of 5.25 million images determined by professional inspectors. The grain samples including wheat, maize and rice are collected from five countries and more than 30 regions. We also develop a comprehensive benchmark based on semi-supervised learning and self-supervised learning techniques. To the best of our knowledge, GrainSpace is the first publicly released dataset for cereal grain inspection, https://github.com/hellodfan/GrainSpace.
</p>
</details>
