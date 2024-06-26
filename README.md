# **【2022 CCF BDCI 基于文心CV大模型的智慧城市视觉多任务识别】第3名方案**

**模型构建及调优**

1.模型总体介绍：

        解题思路：
    
            通过阅读相关文献定位到本任务是一个NAS领域的神经网络模型性能预测问题，但实际上我们并没有获得每个子网络的具体结构，只知道其所在的CV语境下的任务；
    
            此外，我们仅有500条训练数据，也就是说这还是一个小样本学习问题，所以我们考虑使用一些机器学习的手段辅以数据增强方法、以及CV语境先验知识来解决此任务。
    
        算法构建：

            首先，我们通过训练数据重采样，以及根据网络在CV任务中的语义信息，即人脸、人体、车辆和商品识别任务的相关性对各任务单独预测结果进行加权组合生成最终结果来提升模型性能以及泛化性，A榜分数提升至 0.78770625154；

            其次，学习CVPR2022 NAS Track2第一名的解决方案引入Stack Gradient Boosting方法与原模型进行集成综合来提升模型性能，A榜分数提升至 0.79883043865；
    
            再次，根据平台测评结果，从防止模型训练调参过拟合、提升模型泛化性能的角度出发，学习CVPR2022 NAS Track2第四名的解决方案，在部分任务上单独使用CatBoostRanker模型，并调整模型结构编码方式为demo所给出的编码方式以最终提升模型性能，A榜分数提升至 0.79973760966；
            
        代码组织结构：
            
            除GPNAS算法改进源码在另一文件 GPNAS.py 中外，其余代码均在文件 main.ipynb 中。
            
   算法结构框图：
        
   ![](https://ai-studio-static-online.cdn.bcebos.com/d4bab46696324d63bc932bed826634b296b4abf829b54ed7bfc488dc53a0cd7e)

2.数据增强策略：
     
         我们尝试过的数据增强策略有：数据融合、数据重采样、预测数据回填训练。
         
             （1）对于数据融合，我们曾尝试将原训练数据按任务相关性进行加权组合后的结果作为GPNAS模型训练数据，本地效果较差；此外，尝试将GPNAS在各任务单独预测模型中将同类型任务预测结果进行加权组合生成新的结果，测试效果较好，在最终模型5中采用将 veri、vehicleid、veriwild 三个任务单独训练并预测的结果按 0.04、0.92、0.04 的权重组合作为任务 vehicleid 的预测结果；
             
             （2）对于数据重采样，我们认为GPNAS模型在部分任务上初始化较差或欠拟合，故对训练数据进行重采样训练，部分参数下测试效果较好，在最终模型五中 vehicleid 任务的GPNAS模型采取重采样3次训练的策略；
             
             （3）对于预测数据回填训练，我们曾将全部八个任务的模型对于测试集的训练结果进行随机等距采样100条，作为模型的训练数据回填训练，而后再对测试集进行预测，本地尝试效果较好，A榜线上测试效果较差，且随间距选择不同精度波动较大，故暂时搁置，等待日后改进。     

3.调参优化策略：
    
         本地网格搜索交叉验证调参，选取本地测试精度较高若干组参数进行A榜线上测试其泛化能力，选取各任务精度较高模型、参数等进行组合，得到最终模型。
 
4.模型训练以及测试：
           
         详见相关代码块.

5.其它需要说明的内容：
        
         本最终模型由 Sub Model 1、Sub Model 2、Sub Model 3 组合而成；
             
         其中 Sub Model 1 用于任务 0、1、3、7，此模型源自 CVPR 2022 NAS Track2 Rank 1 解决方案，我们在其基础上进行了一些改进，原项目链接如下 https://aistudio.baidu.com/aistudio/projectdetail/3751972?channel=0&channelType=0&sUid=2709743&shared=1&ts=1668438343541
             
         其中 Sub Model 2 用于任务 2、4、6，此模型源自 CVPR 2022 NAS Track2 Rank 4 解决方案，我们在其基础上进行了一些改进，项目链接如下 https://aistudio.baidu.com/aistudio/clusterprojectdetail/4051842
             
         其中 Sub Model 3 用于任务 5， 此模型为对官方基线demo以及GPNAS方法的改进。
 
 6.特别注意：
         
         本团队所提交的最高得分结果中，产生任务 0、1、3、7 预测结果的 Sub Model 1 运行环境为 PaddlePaddle 2.2.2，本项目基于 PaddlePaddle 2.3.2，在此环境下模型经训练并测试所产生的结果略有不用，若想完全复现本团队最高得分结果，请修改项目环境为 PaddlePaddle 2.2.2，以获取对应任务预测结果并与其他任务预测结果进行组合；
         
         此外，其余产生任务 2、4、5、6 预测结果的模型运行环境均为 PaddlePaddle 2.3.2，经本团队测试，其在 PaddlePaddle 2.2.2 环境下运行结果亦会产生差别，请特别注意；
         
         总结而言，若想完全复现本团队最高分结果，请在 PaddlePaddle 2.2.2 对任务 0、1、3、7 进行预测，在 PaddlePaddle 2.3.2 对任务 2、4、5、6 进行预测，而后将结果进行组合，本项目基于 PaddlePadle 2.3.2，若预测结果经B榜测试分数有差异且会造成名次影响，请按上述流程进行重新获取结果，敬请谅解；
         
         本项目只保存了 PaddlePaddle 2.3.2 环境下模型 Checkpoint 文件。
