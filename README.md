### 1.SLPA
算法介绍： [SLPA](https://blog.csdn.net/u010159842/article/details/100217337/?ops_request_misc=&request_id=&biz_id=102&utm_term=%E6%A0%87%E7%AD%BE%E4%BC%A0%E6%92%AD%E6%8C%87%E6%A0%87&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-100217337.142^v100^pc_search_result_base9&spm=1018.2226.3001.4187)

### 2. data\_process/find\_nodes
通过两个边数据集确定0，1，2层节点，对于没有tag的赋空值即可。demo是取13个0层节点+每个0层节点取100个1层节点+每个1层节点最后取5个2层节点，但最后实际取得1389个节点（因为这100个1层节点并不一定每个都有5个粉丝）
### 3. run\_SLPA
对用户标签进行了切割，一部分用于传播，一部分用于验证。最后利用传播得到的标签与验证标签计算precision、recall、f1,但目前三个指标均为0，一是因为用户tag自由度极高，二是因为这些用户与他们的粉丝的关联性不是很大。

