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
<a href="https://ieeexplore.ieee.org/document/9156329">
BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning
</a></summary>
<p>
Datasets drive vision progress, yet existing driving datasets are impoverished in terms of visual content and supported tasks to study multitask learning for autonomous driving. Researchers are usually constrained to study a small set of problems on one dataset, while real-world computer vision applications require performing tasks of various complexities. We construct BDD100K, the largest driving video dataset with 100K videos and 10 tasks to evaluate the exciting progress of image recognition algorithms on autonomous driving. The dataset possesses geographic, environmental, and weather diversity, which is useful for training models that are less likely to be surprised by new conditions. Based on this diverse dataset, we build a benchmark for heterogeneous multitask learning and study how to solve the tasks together. Our experiments show that special training strategies are needed for existing models to perform such heterogeneous tasks. BDD100K opens the door for future studies in this important venue.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/9879019">
AutoMine: An Unmanned Mine Dataset
</a></summary>
<p>
Autonomous driving datasets have played an important role in validating the advancement of intelligent vehicle algorithms including localization, perception and prediction in academic areas. However, current existing datasets pay more attention to the structured urban road, which hampers the exploration on unstructured special scenarios. Moreover, the open-pit mine is one of the typical representatives for them. Therefore, we introduce the Autonomous driving dataset on the Mining scene (AutoMine) for positioning and perception tasks in this paper. The AutoMine is collected by multiple acquisition platforms including an SUV, a wide-body mining truck and an ordinary mining truck, depending on the actual mine operation scenarios. The dataset consists of 18+ driving hours, 18K annotated lidar and image frames for 3D perception with various mines, time-of-the-day and weather conditions. The main contributions of the AutoMine dataset are as follows: I.The first autonomous driving dataset for perception and localization in mine scenarios. 2.There are abundant dynamic obstacles of 9 degrees of freedom with large dimension difference (mining trucks and pedestrians) and extreme climatic conditions (the dust and snow) in the mining area. 3.Multi-platform acquisition strategies could capture mining data from multiple perspectives that fit the actual operation. More details can be found in our website(https://automine.cc).
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/9157053">
Google Landmarks Dataset v2 – A Large-Scale Benchmark for Instance-Level Recognition and Retrieval
</a></summary>
<p>
While image retrieval and instance recognition techniques are progressing rapidly, there is a need for challenging datasets to accurately measure their performance -- while posing novel challenges that are relevant for practical applications. We introduce the Google Landmarks Dataset v2 (GLDv2), a new benchmark for large-scale, fine-grained instance recognition and image retrieval in the domain of human-made and natural landmarks. GLDv2 is the largest such dataset to date by a large margin, including over 5M images and 200k distinct instance labels. Its test set consists of 118k images with ground truth annotations for both the retrieval and recognition tasks. The ground truth construction involved over 800 hours of human annotator work. Our new dataset has several challenging properties inspired by real-world applications that previous datasets did not consider: An extremely long-tailed class distribution, a large fraction of out-of-domain test photos and large intra-class variability. The dataset is sourced from Wikimedia Commons, the world's largest crowdsourced collection of landmark photos. We provide baseline results for both recognition and retrieval tasks based on state-of-the-art methods as well as competitive results from a public challenge. We further demonstrate the suitability of the dataset for transfer learning by showing that image embeddings trained on it achieve competitive retrieval performance on independent datasets. The dataset images, ground-truth and metric scoring code are available at https://github.com/cvdfoundation/google-landmark
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/10205454">
Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method
</a></summary>
<p>
Image aesthetics assessment (IAA) is a challenging task due to its highly subjective nature. Most of the current studies rely on large-scale datasets (e.g., AVA and AADB) to learn a general model for all kinds of photography images. However, little light has been shed on measuring the aesthetic quality of artistic images, and the existing datasets only contain relatively few artworks. Such a defect is a great obstacle to the aesthetic assessment of artistic images. To fill the gap in the field of artistic image aesthetics assessment (AIAA), we first introduce a large-scale AIAA dataset: Boldbrush Artistic Image Dataset (BAlD), which consists of 60,337 artistic images covering various art forms, with more than 360,000 votes from online users. We then propose a new method, SAAN (Style-specific Art Assessment Network), which can effectively extract and utilize style-specific and generic aesthetic information to evaluate artistic images. Experiments demonstrate that our proposed approach outperforms existing lAA methods on the proposed BAlD dataset according to quantitative comparisons. We believe the proposed dataset and method can serve as a foundation for future AIAA works and inspire more research in this field. Dataset and code are available at: https://github.com/Dreemurr-T/BAID.git
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/10656008">
Towards Modern Image Manipulation Localization: A Large-Scale Dataset and Novel Methods
</a></summary>
<p>
In recent years, image manipulation localization has attracted increasing attention due to its pivotal role in guaranteeing social media security. However, how to accurately identify the forged regions remains an open challenge. One of the main bottlenecks lies in the severe scarcity of high-quality data, due to its costly creation process. To address this limitation, we propose a novel paradigm, termed as CAAA, to automatically and precisely annotate the numerous manually forged images from the web at the pixel level. We further propose a novel metric QES to facilitate the automatic filtering of unreliable annotations. With CAAA and QES, we construct a large-scale, diverse, and high-quality dataset comprising 123,150 manually forged images with mask annotations. Besides, we develop a new model APSC-Net for accurate image manipulation localization. According to extensive experiments, our dataset significantly improves the performance of various models on the widely-used benchmarks and such improvements are attributed to our proposed effective methods. The dataset and code are publicly available at https://github.com/qcf-568/MIML.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/10204595">
JRDB-Pose: A Large-Scale Dataset for Multi-Person Pose Estimation and Tracking
</a></summary>
<p>
Autonomous robotic systems operating in human environments must understand their surroundings to make accurate and safe decisions. In crowded human scenes with close-up human-robot interaction and robot navigation, a deep understanding of surrounding people requires reasoning about human motion and body dynamics over time with human body pose estimation and tracking. However, existing datasets captured from robot platforms either do not provide pose annotations or do not reflect the scene distribution of social robots. In this paper, we introduce JRDB-Pose, a large-scale dataset and benchmark for multi-person pose estimation and tracking. JRDB-Pose extends the existing JRDB which includes videos captured from a social navigation robot in a university campus environment, containing challenging scenes with crowded indoor and outdoor locations and a diverse range of scales and occlusion types. JRDB-Pose provides human pose annotations with per-keypoint occlusion labels and track IDs consistent across the scene and with existing annotations in JRDB. We conduct a thorough experimental study of state-of-the-art multi-person pose estimation and tracking methods on JRDB-Pose, showing that our dataset imposes new challenges for the existing methods. JRDB-Pose is available at https://jrdb.erc.monash.edu/.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/9156686">
DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection
</a></summary>
<p>
We present our on-going effort of constructing a large- scale benchmark for face forgery detection. The first version of this benchmark, DeeperForensics-1.0, represents the largest face forgery detection dataset by far, with 60, 000 videos constituted by a total of 17.6 million frames, 10 times larger than existing datasets of the same kind. Extensive real-world perturbations are applied to obtain a more challenging benchmark of larger scale and higher diversity. All source videos in DeeperForensics-1.0 are carefully collected, and fake videos are generated by a newly proposed end-to-end face swapping framework. The quality of generated videos outperforms those in existing datasets, validated by user studies. The benchmark features a hidden test set, which contains manipulated videos achieving high deceptive scores in human evaluations. We further contribute a comprehensive study that evaluates five representative detection baselines and make a thorough analysis of different settings.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/5995586">
A large-scale benchmark dataset for event recognition in surveillance video
</a></summary>
<p>
We introduce a new large-scale video dataset designed to assess the performance of diverse visual event recognition algorithms with a focus on continuous visual event recognition (CVER) in outdoor areas with wide coverage. Previous datasets for action recognition are unrealistic for real-world surveillance because they consist of short clips showing one action by one individual [15, 8]. Datasets have been developed for movies [11] and sports [12], but, these actions and scene conditions do not apply effectively to surveillance videos. Our dataset consists of many outdoor scenes with actions occurring naturally by non-actors in continuously captured videos of the real world. The dataset includes large numbers of instances for 23 event types distributed throughout 29 hours of video. This data is accompanied by detailed annotations which include both moving object tracks and event examples, which will provide solid basis for large-scale evaluation. Additionally, we propose different types of evaluation modes for visual recognition tasks and evaluation metrics along with our preliminary experimental results. We believe that this dataset will stimulate diverse aspects of computer vision research and help us to advance the CVER tasks in the years ahead.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/9156308">
AnimalWeb: A Large-Scale Hierarchical Dataset of Annotated Animal Faces
</a></summary>
<p>
Several studies show that animal needs are often expressed through their faces. Though remarkable progress has been made towards the automatic understanding of human faces, this has not been the case with animal faces. There exists significant room for algorithmic advances that could realize automatic systems for interpreting animal faces. Besides scientific value, resulting technology will foster better and cheaper animal care. We believe the underlying research progress is mainly obstructed by the lack of an adequately annotated dataset of animal faces, covering a wide spectrum of animal species. To this end, we introduce a large-scale, hierarchical annotated dataset of animal faces, featuring 22.4K faces from 350 diverse species and 21 animal orders across biological taxonomy. These faces are captured `in-the-wild' conditions and are consistently annotated with 9 landmarks on key facial features. The dataset is structured and scalable by design; its development underwent four systematic stages involving rigorous, overall effort of over 6K man-hours. We benchmark it for face alignment using the existing art under two new problem settings. Results showcase its challenging nature, unique attributes and present definite prospects for novel, adaptive, and generalized face-oriented CV algorithms. Further benchmarking the dataset across face detection and fine-grained recognition tasks demonstrates its multi-task applications and room for improvement. The dataset is available at: https://fdmaproject.wordpress.com/.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/11094675">
NSD-Imagery: A benchmark dataset for extending fMRI vision decoding methods to mental imagery
</a></summary>
<p>
We release NSD-Imagery, a benchmark dataset of human fMRI activity paired with mental images, to complement the existing Natural Scenes Dataset (NSD), a large-scale dataset of fMRI activity paired with seen images that enabled unprecedented improvements in fMRI-to-image reconstruction efforts. Recent models trained on NSD have been evaluated only on seen image reconstruction. Using NSD-Imagery, it is possible to assess how well these models perform on mental image reconstruction. This is a challenging generalization requirement because mental images are encoded in human brain activity with relatively lower signal-to-noise and spatial resolution; however, generalization from seen to mental imagery is critical for real-world applications in medical domains and brain-computer interfaces, where the desired information is always internally generated. We provide benchmarks for a suite of recent NSD-trained open-source visual decoding models (MindEye1, MindEye2, Brain Diffuser, iCNN, Takagi et al.) on NSD-Imagery, and show that the performance of decoding methods on mental images is largely decoupled from performance on vision reconstruction. We further demonstrate that architectural choices significantly impact cross-decoding performance: models employing simple linear decoding architectures and multi-modal feature decoding generalize better to mental imagery, while complex architectures tend to overfit visual training data. Our findings indicate that mental imagery datasets are critical for the development of practical applications, and establish NSD-Imagery as a useful resource for better aligning visual decoding methods with this goal.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/10658200">
Real-IAD: A Real-World Multi-View Dataset for Benchmarking Versatile Industrial Anomaly Detection
</a></summary>
<p>
Industrial anomaly detection (I AD) has garnered signif-icant attention and experienced rapid development. However, the recent development of I AD approach has encountered certain difficulties due to dataset limitations. On the one hand, most of the state-of-the-art methods have achieved saturation (over 99% in AUROC) on mainstream datasets such as MVTec, and the differences of methods cannot be well distinguished, leading to a significant gap between public datasets and actual application scenarios. On the other hand, the research on various new practical anomaly detection settings is limited by the scale of the dataset, posing a risk of overfitting in evaluation results. Therefore, we propose a large-scale, Real-world, and multi-view Industrial Anomaly Detection dataset, named Real- I AD, which contains 150K high-resolution images of 30 different objects, an order of magnitude larger than existing datasets. It has a larger range of defect area and ratio proportions, making it more challenging than previous datasets. To make the dataset closer to real application scenarios, we adopted a multi-view shooting method and proposed sample-level evaluation metrics. In addition, beyond the general unsupervised anomaly detection setting, we propose a new setting for Fully Unsupervised Indus-trial Anomaly Detection (FUIAD) based on the observation that the yield rate in industrial production is usually greater than 60%, which has more practical application value. Finally, we report the results of popular I AD methods on the Real- I AD dataset, providing a highly challenging benchmark to promote the development of the I AD field.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/10658179">
TUMTraf V2X Cooperative Perception Dataset
</a></summary>
<p>
Cooperative perception offers several benefits for en-hancing the capabilities of autonomous vehicles and im-proving road safety. Using roadside sensors in addition to onboard sensors increases reliability and extends the sensor range. External sensors offer higher situational awareness for automated vehicles and prevent occlusions. We propose CoopDet3D, a cooperative multi-modal fusion model, and TUMTraf- V2X, a perception dataset, for the cooperative 3D object detection and tracking task. Our dataset contains 2,000 labeled point clouds and 5,000 labeled images from five roadside and four onboard sensors. It includes 30k 3D boxes with track IDs and precise GPS and IMU data. We labeled nine categories and covered occlusion scenarios with challenging driving maneuvers, like traffic violations, near-miss events, overtaking, and U-turns. Through multiple experiments, we show that our CoopDet3D camera-LiDARfusion model achieves an increase of +14.36 3D mAP compared to a vehicle camera-LiDARfusion model. Finally, we make our dataset, model, labeling tool, and devkit publicly available on our website.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/9880410">
GrainSpace: A Large-scale Dataset for Fine-grained and Domain-adaptive Recognition of Cereal Grains
</a></summary>
<p>
Cereal grains are a vital part of human diets and are important commodities for people's livelihood and international trade. Grain Appearance Inspection (GAI) serves as one of the crucial steps for the determination of grain quality and grain stratification for proper circulation, storage and food processing, etc. GAI is routinely performed manually by qualified inspectors with the aid of some hand tools. Automated GAI has the benefit of greatly assisting inspectors with their jobs but has been limited due to the lack of datasets and clear definitions of the tasks. In this paper we formulate GAI as three ubiquitous computer vision tasks: fine-grained recognition, domain adaptation and out-of-distribution recognition. We present a large-scale and publicly available cereal grains dataset called GrainSpace. Specifically, we construct three types of device prototypes for data acquisition, and a total of 5.25 million images determined by professional inspectors. The grain samples including wheat, maize and rice are collected from five countries and more than 30 regions. We also develop a comprehensive benchmark based on semi-supervised learning and self-supervised learning techniques. To the best of our knowledge, GrainSpace is the first publicly released dataset for cereal grain inspection, https://github.com/hellodfan/GrainSpace.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/9879063">
ZeroWaste Dataset: Towards Deformable Object Segmentation in Cluttered Scenes
</a></summary>
<p>
Less than 35% of recyclable waste is being actually recycled in the US [2], which leads to increased soil and sea pollution and is one of the major concerns of environmental researchers as well as the common public. At the heart of the problem are the inefficiencies of the waste sorting process (separating paper, plastic, metal, glass, etc.) due to the extremely complex and cluttered nature of the waste stream. Recyclable waste detection poses a unique computer vision challenge as it requires detection of highly deformable and often translucent objects in cluttered scenes without the kind of context information usually present in human-centric datasets. This challenging computer vision task currently lacks suitable datasets or methods in the available literature. In this paper, we take a step towards computer-aided waste detection and present the first in-the-wild industrial-grade waste detection and segmentation dataset, ZeroWaste. We believe that ZeroWaste will catalyze research in object detection and semantic segmentation in extreme clutter as well as applications in the recycling domain. Our project page can be found at http://ai.bu.edu/zerowaste/
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/9878471">
FERV39k: A Large-Scale Multi-Scene Dataset for Facial Expression Recognition in Videos
</a></summary>
<p>
Current benchmarks for facial expression recognition (FER) mainly focus on static images, while there are limited datasets for FER in videos. It is still ambiguous to evaluate whether performances of existing methods remain satisfactory in real-world application-oriented scenes. For example, the “Happy” expression with high intensity in Talk-Show is more discriminating than the same expression with low intensity in Official-Event. To fill this gap, we build a large-scale multi-scene dataset, coined as FERV39k. We analyze the important ingredients of constructing such a novel dataset in three aspects: (1) multi-scene hierarchy and expression class, (2) generation of candidate video clips, (3) trusted manual labelling process. Based on these guidelines, we select 4 scenarios subdivided into 22 scenes, annotate 86k samples automatically obtained from 4k videos based on the well-designed workflow, and finally build 38,935 video clips labeled with 7 classic expressions. Experiment benchmarks on four kinds of baseline frame-works were also provided and further analysis on their performance across different scenes and some challenges for future research were given. Besides, we systematically investigate key components of DFER by ablation studies. The baseline framework and our project are available on https://github.com/wangyanckxx/FERV39k.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/8954457">
LVIS: A Dataset for Large Vocabulary Instance Segmentation
</a></summary>
<p>
Progress on object detection is enabled by datasets that focus the research community’s attention on open challenges. This process led us from simple images to complex scenes and from bounding boxes to segmentation masks. In this work, we introduce LVIS (pronounced ‘el-vis’): a new dataset for Large Vocabulary Instance Segmentation. We plan to collect 2.2 million high-quality instance segmentation masks for over 1000 entry-level object categories in 164k images. Due to the Zipfian distribution of categories in natural images, LVIS naturally has a long tail of categories with few training samples. Given that state-of-the-art deep learning methods for object detection perform poorly in the low-sample regime, we believe that our dataset poses an important and exciting new scientific challenge. LVIS is available at http://www.lvisdataset.org.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/10205154">
MVImgNet: A Large-scale Dataset of Multi-view Images
</a></summary>
<p>
Being data-driven is one of the most iconic properties of deep learning algorithms. The birth of ImageNet [24] drives a remarkable trend of ‘learning from large-scale data’ in computer vision. Pretraining on ImageNet to obtain rich universal representations has been manifested to benefit various 2D visual tasks, and becomes a standard in 2D vision. However, due to the laborious collection of real-world 3D data, there is yet no generic dataset serving as a counterpart of ImageNet in 3D vision, thus how such a dataset can impact the 3D community is unraveled. To remedy this defect, we introduce MVImgNet, a large-scale dataset of multi-view images, which is highly convenient to gain by shooting videos of real-world objects in human daily life. It contains 6.5 million frames from 219,188 videos crossing objects from 238 classes, with rich annotations of object masks, camera parameters, and point clouds. The multi-view attribute endows our dataset with 3D-aware signals, making it a soft bridge between 2D and 3D vision. We conduct pilot studies for probing the potential of MVImgNet on a variety of 3D and 2D visual tasks, including radiance field reconstruction, multi-view stereo, and view-consistent image understanding, where MVImgNet demonstrates promising performance, remaining lots of possibilities for future explorations. Besides, via dense reconstruction on MVImgNet, a 3D object point cloud dataset is derived, called MVPNet, covering 87,200 samples from 150 categories, with the class label on each point cloud. Experiments show that MVP-Net can benefit the real-world 3D object classification while posing new challenges to point cloud understanding. MVImgNet and MVPNet will be public, hoping to inspire the broader vision community.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/10655889">
Insect-Foundation: A Foundation Model and Large-Scale 1M Dataset for Visual Insect Understanding
</a></summary>
<p>
In precision agriculture, the detection and recognition of insects play an essential role in the ability of crops to grow healthy and produce a high-quality yield. The current machine vision model requires a large volume of data to achieve high performance. However, there are approximately 5.5 million different insect species in the world. None of the existing insect datasets can cover even a fraction of them due to varying geographic locations and acquisition costs. In this paper, we introduce a novel “Insect-1M” dataset, a game-changing resource poised to revolutionize insect-related foundation model training. Covering a vast spectrum of insect species, our dataset, including 1 million images with dense identification labels of taxonomy hierarchy and insect descriptions, offers a panoramic view of entomology, enabling foundation models to comprehend visual and semantic information about insects like never before. Then, to efficiently establish an Insect Foundation Model, we develop a micro-feature self-supervised learning method with a Patch-wise Relevant Attention mechanism capable of discerning the subtle differences among insect images. In addition, we introduce Description Consistency loss to improve micro-feature modeling via insect descriptions. Through our experiments, we illustrate the effectiveness of our proposed approach in insect modeling and achieve State-of-the-Art performance on standard benchmarks of insect-related tasks. Our Insect Foundation Model and Dataset promise to empower the next generation of insect-related vision models, bringing them closer to the ultimate goal of precision agriculture.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/11092891">
ORIDa: Object-centric Real-world Image Composition Dataset
</a></summary>
<p>
Object compositing, the task of placing and harmonizing objects in images of diverse visual scenes, has become an important task in computer vision with the rise of generative models. However, existing datasets lack the diversity and scale required to comprehensively explore real-world scenarios. We introduce ORIDa (Object-centric Real-world Image Composition Dataset), a large-scale, real-captured dataset containing over 30,000 images featuring 200 unique objects, each of which is presented across varied positions and scenes. ORIDa has two types of data: factual-counterfactual sets and factual-only scenes. The factual-counterfactual sets consist of four factual images showing an object in different positions within a scene and a single counterfactual (or background) image of the scene without the object, resulting in five images per scene. The factual-only scenes include a single image containing an object in a specific context, expanding the variety of environments. To our knowledge, ORIDa is the first publicly available dataset with its scale and complexity for real-world image composition. Extensive analysis and experiments highlight the value of ORIDa as a resource for advancing further research in object compositing.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/10204878">
V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting
</a></summary>
<p>
Utilizing infrastructure and vehicle-side information to track and forecast the behaviors of surrounding traffic participants can significantly improve decision-making and safety in autonomous driving. However, the lack of real-world sequential datasets limits research in this area. To address this issue, we introduce V2X-Seq, the first large-scale sequential V2X dataset, which includes data frames, trajectories, vector maps, and traffic lights captured from natural scenery. V2X-Seq comprises two parts: the sequential perception dataset, which includes more than 15,000 frames captured from 95 scenarios, and the trajectory forecasting dataset, which contains about 80,000 infrastructure-view scenarios, 80,000 vehicle-view scenarios, and 50,000 cooperative-view scenarios captured from 28 intersections' areas, covering 672 hours of data. Based on V2X-Seq, we introduce three new tasks for vehicle-infrastructure cooperative (VIC) autonomous driving: VIC3D Tracking, Online-VIC Forecasting, and Offline-VIC Forecasting. We also provide benchmarks for the introduced tasks. Find data, code, and more up-to-date information at https://github.com/AIR-THU/DAIR-V2X-Seq.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/10204714">
DF-Platter: Multi-Face Heterogeneous Deepfake Dataset
</a></summary>
<p>
Deepfake detection is gaining significant importance in the research community. While most of the research efforts are focused towards high-quality images and videos with controlled appearance of individuals, deepfake generation algorithms now have the capability to generate deep-fakes with low-resolution, occlusion, and manipulation of multiple subjects. In this research, we emulate the real-world scenario of deepfake generation and propose the DF-Platter dataset, which contains (i) both low-resolution and high-resolution deepfakes generated using multiple generation techniques and (ii) single-subject and multiple-subject deepfakes, with face images of Indian ethnicity. Faces in the dataset are annotated for various attributes such as gender, age, skin tone, and occlusion. The dataset is prepared in 116 days with continuous usage of 32 GPUs accounting to 1,800 GB cumulative memory. With over 500 GBs in size, the dataset contains a total of 133,260 videos encompassing three sets. To the best of our knowledge, this is one of the largest datasets containing vast variability and multiple challenges. We also provide benchmark results under multiple evaluation settings using popular and state-of-the-art deepfake detection models, for c0 images and videos along with c23 and c40 compression variants. The results demonstrate a significant performance reduction in the deepfake detection task on low-resolution deep-fakes. Furthermore, existing techniques yield declined detection accuracy on multiple-subject deepfakes. It is our assertion that this database will improve the state-of-the-art by extending the capabilities of deepfake detection algorithms to real-world scenarios. The database is available at: http://iab-rubric.org/df-platter-database.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/9156681">
FSS-1000: A 1000-Class Dataset for Few-Shot Segmentation
</a></summary>
<p>
Over the past few years, we have witnessed the success of deep learning in image recognition thanks to the availability of large-scale human-annotated datasets such as PASCAL VOC, ImageNet, and COCO. Although these datasets have covered a wide range of object categories, there are still a significant number of objects that are not included. Can we perform the same task without a lot of human annotations? In this paper, we are interested in few-shot object segmentation where the number of annotated training examples are limited to 5 only. To evaluate and validate the performance of our approach, we have built a few-shot segmentation dataset, FSS-1000, which consists of 1000 object classes with pixelwise annotation of ground-truth segmentation. Unique in FSS-1000, our dataset contains significant number of objects that have never been seen or annotated in previous datasets, such as tiny daily objects, merchandise, cartoon characters, logos, etc. We build our baseline model using standard backbone networks such as VGG-16, ResNet-101, and Inception. To our surprise, we found that training our model from scratch using FSS-1000 achieves comparable and even better results than training with weights pre-trained by ImageNet which is more than 100 times larger than FSS-1000. Both our approach and dataset are simple, effective, and easily extensible to learn segmentation of new object classes given very few annotated training examples. Dataset is available at https://github.com/HKUSTCV/FSS-1000
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/8954351">
IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition
</a></summary>
<p>
Insect pests are one of the main factors affecting agricultural product yield. Accurate recognition of insect pests facilitates timely preventive measures to avoid economic losses. However, the existing datasets for the visual classification task mainly focus on common objects, e.g., flowers and dogs. This limits the application of powerful deep learning technology on specific domains like the agricultural field. In this paper, we collect a large-scale dataset named IP102 for insect pest recognition. Specifically, it contains more than 75, 000 images belonging to 102 categories, which exhibit a natural long-tailed distribution. In addition, we annotate about 19, 000 images with bounding boxes for object detection. The IP102 has a hierarchical taxonomy and the insect pests which mainly affect one specific agricultural product are grouped into the same upperlevel category. Furthermore, we perform several baseline experiments on the IP102 dataset, including handcrafted and deep feature based classification methods. Experimental results show that this dataset has the challenges of interand intra- class variance and data imbalance. We believe our IP102 will facilitate future research on practical insect pest control, fine-grained visual classification, and imbalanced learning fields. We make the dataset and pre-trained models publicly available at https://github.com/xpwu95/IP102.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/11094424">
OpenHumanVid: A Large-Scale High-Quality Dataset for Enhancing Human-Centric Video Generation
</a></summary>
<p>
Recent advancements in visual generation technologies have markedly increased the scale and availability of video datasets, which are crucial for training effective video generation models. However, a significant lack of high-quality, human-centric video datasets presents a challenge to progress in this field. To bridge this gap, we introduce OpenHumanVid, a large-scale and high-quality humancentric video dataset characterized by precise and detailed captions that encompass both human appearance and motion states, along with supplementary human motion conditions, including skeleton sequences and speech audio. To validate the efficacy of this dataset and the associated training strategies, we propose an extension of existing classical diffusion transformer architectures and conduct further pretraining of our models on the proposed dataset. Our findings yield two critical insights: First, the incorporation of a large-scale, high-quality dataset substantially enhances evaluation metrics for generated human videos while preserving performance in general video generation tasks. Second, the effective alignment of text with human appearance, human motion, and facial motion is essential for producing high-quality video outputs. Based on these insights and corresponding methodologies, the straightforward extended network trained on the proposed dataset demonstrates an obvious improvement in the generation of human-centric videos. The source code and the dataset are available at: https://fudan-generative-vision.github.io/OpenHumanVid.
</p>
</details>
<details>
<summary>
<a href="https://ieeexplore.ieee.org/document/11094586">
Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline
</a></summary>
<p>
Interactive Medical Image Segmentation (IMIS) has long been constrained by the limited availability of large-scale, diverse, and densely annotated datasets, which hinders model generalization and consistent evaluation across different models. In this paper, we introduce the IMed-361M benchmark dataset, a significant advancement in general IMIS research. First, we collect and standardize over 6.4 million medical images and their corresponding ground truth masks from multiple data sources. Then, leveraging the strong object recognition capabilities of a vision foundational model, we automatically generated dense interactive masks for each image and ensured their quality through rigorous quality control and granularity management. Unlike previous datasets, which are limited by specific modalities or sparse annotations, IMed-361M spans 14 modalities and 204 segmentation targets, totaling 361 million masks—an average of 56 masks per image. Finally, we developed an IMIS baseline network on this dataset that supports high-quality mask generation through interactive inputs, including clicks, bounding boxes, text prompts, and their combinations. We evaluate its performance on medical image segmentation tasks from multiple perspectives, demonstrating superior accuracy and scalability compared to existing interactive segmentation models. To facilitate research on foundational models in medical computer vision, we release the IMed-361M and model at https://github.com/uni-medical/IMIS-Bench.
</p>
</details>
