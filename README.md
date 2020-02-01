Capacity, Bandwidth, and Compositionality in Emergent Language Learning
==================================
Code repository of the models described in the paper accepted at AAMAS 2020 
[Capacity, Bandwidth, and Compositionality in Emergent Language Learning](http://www.google.com "Capacity, Bandwidth, and Compositionality in Emergent Language Learning").

Dependencies
------------------
### Python
* Python>=3.6
* PyTorch>=1.2

### GPU
* CUDA>=10.1
* cuDNN>=7.6

Running code
------------------
```bash
$ python main.py --num-binary-messages 24 --num-digits 6 --embedding-size-sender 40 --project-size-sender 60 --num-lstm-sender 300 --num-lstm-receiver 325 --embedding-size-receiver 125 --save-str <SAVE_STR>
```
where `num-binary-messages` is the bandwidth, `num-digits` is the number of concepts, and `<SAVE_STR>` is the filename.

License
-------------------
This project is licensed under the terms of the MIT license.


Citation
------------------
If you find the resources in this repository useful, please consider citing:
```
@inproceedings{resnick*2020cap,
    author = {Resnick*, Cinjon and Gupta*, Abhinav and Foerster, Jakob and Dai, Andrew and Cho, Kyunghyun},
    title = {Capacity, Bandwidth, and Compositionality in Emergent Language Learning},
    year = {2020},
    publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
    address = {Richland, SC},
    booktitle = {Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems},
    numpages = {9},
    keywords = {Multi-agent communication, Compositionality, Emergent languages},
    location = {Auckland, New Zealand},
    series = {AAMAS ’20},
    url = {https://arxiv.org/abs/1910.11424},
}
```