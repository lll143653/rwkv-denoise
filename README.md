基于RWKV6，借鉴了[lll143653/SocketIOClient-Unreal](https://github.com/OpenGVLab/Vision-RWKV)和https://github.com/feizc/Diffusion-RWKV的工作整合（缝合）完成，基于SIDD-Medium训练得到，在sidd validation上效果为39.34db。

## how to inference
运行
`
 python inference.py --sidd /mnt/f/datasets/SIDD 
`
可对sidd验证集进行测试。sidd 验证集可从https://abdokamel.github.io/sidd/#sidd-medium下载
运行
`
 python inference.py --img weights/0.png
`
可对单张图片进行去噪。