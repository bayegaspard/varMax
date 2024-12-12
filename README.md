# varMax
Towards Confidence-Based Zero-Day Attack Recognition
### Abstract
Detecting zero-day attacks, which exploit previously unknown vulnerabilities, is vital, especially in mission-critical systems. For such attacks, using Deep Neural Networks (DNNs) for detection is often ineffective because they usually make overly confident and incorrect predictions when faced with new types of attacks. This problem stems from the fundamental design of the softmax function used in DNNs, which is good at identifying familiar attack types but struggles with recognizing new, unknown ones. Previous research has shown that open-set recognition (OSR) algorithms that are created to enable DNN models to differentiate between known and unknown inputs tend to show a bias towards flagging inputs as unknown, indicating a need for a more balanced approach. To address this gap, this paper introduces varMax, a novel, bias-neutral technique for OSR that utilizes the variance in DNN logits to distinguish between known and unknown inputs. It comprises three key components: (1) a top-difference algorithm that assesses class certainty by comparing the top two softmax scores against a predetermined threshold; (2) a method for categorizing ambiguous samples as known or unknown based on logit variance in the final DNN layer; (3) an adapted energy-based out-of-distribution function to boost the accuracy and trustworthiness of classifications. Our extensive evaluations demonstrate that varMax surpasses existing leading methods in effectively identifying unknown activities, while also enhancing the DNN's confidence and robustness in distinguishing between known and unknown inputs, or zero-day attacks. This research marks a significant step forward in the development of more reliable and unbiased intrusion detection systems for cybersecurity threats.

If you want to cite; use the following :
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
