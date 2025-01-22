# varMax
Towards Confidence-Based Zero-Day Attack Recognition
Official repository for the paper implementation of varMax published at the IEEE MILCOM 2024 Track 5 - Machine Learning for Communications and Networking.
### Abstract
Detecting zero-day attacks, which exploit previously unknown vulnerabilities, is vital, especially in mission-critical systems. For such attacks, using Deep Neural Networks (DNNs) for detection is often ineffective because they usually make overly confident and incorrect predictions when faced with new types of attacks. This problem stems from the fundamental design of the softmax function used in DNNs, which is good at identifying familiar attack types but struggles with recognizing new, unknown ones. Previous research has shown that open-set recognition (OSR) algorithms that are created to enable DNN models to differentiate between known and unknown inputs tend to show a bias towards flagging inputs as unknown, indicating a need for a more balanced approach. To address this gap, this paper introduces varMax, a novel, bias-neutral technique for OSR that utilizes the variance in DNN logits to distinguish between known and unknown inputs. It comprises three key components: (1) a top-difference algorithm that assesses class certainty by comparing the top two softmax scores against a predetermined threshold; (2) a method for categorizing ambiguous samples as known or unknown based on logit variance in the final DNN layer; (3) an adapted energy-based out-of-distribution function to boost the accuracy and trustworthiness of classifications. Our extensive evaluations demonstrate that varMax surpasses existing leading methods in effectively identifying unknown activities, while also enhancing the DNN's confidence and robustness in distinguishing between known and unknown inputs, or zero-day attacks. This research marks a significant step forward in the development of more reliable and unbiased intrusion detection systems for cybersecurity threats.

### Installation
```
$ git clone https://github.com/bayegaspard/varMax.git
$ cd varMax
$ pip install -r requirements.txt
```
### Datasets
Refer to : [Payload-Byte](https://github.com/Yasir-ali-farrukh/Payload-Byte.git)


### Architecture of varMax

<img 
 style="text-align: center;"
    src="https://github.com/user-attachments/assets/56d56315-f0f2-496e-9952-0abc6d8587d9">
</img>


Cite
```
@inproceedings{baye2024varmax,
  title={varMax: Towards Confidence-Based Zero-Day Attack Recognition},
  author={Baye, Gaspard and Silva, Priscila and Broggi, Alexandre and Bastian, Nathaniel D and Fiondella, Lance and Kul, Gokhan},
  booktitle={MILCOM 2024-2024 IEEE Military Communications Conference (MILCOM)},
  pages={863--868},
  year={2024},
  organization={IEEE}
}
```
### Acknowledgement
> This work has been funded by UMass Dartmouth and was supported by the U.S. Military Academy (USMA) under Cooperative Agreement No. W911NF-22- 2-0160. The views and conclusions expressed in this paper are those of the authors and do not reflect the official policy or position of the U.S. Military Academy, U.S. Army, U.S. Department of Homeland Security, or U.S. Government. Usual disclaimers apply.
