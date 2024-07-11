# MultiSmell

## abstract
Code smells are those bad code structures that violate design principles and make software systems difficult to understand and maintain. Although many techniques have been proposed, most approaches only detect one kind of code smell and fail to identify multiple smells simultaneously. Furthermore, existing multi-label detection approaches, e.g. the classifier chain, suffer from propagation errors along the chain. To this end, this paper proposes MultiSmell, a novel approach to detect multiple code smells simultaneously based on a heterogeneous graph. Firstly, we select 28 real-world projects as a corpus to construct the dataset. Both textual and metric features are extracted from these projects to generate the samples labeled with multiple code smells by static analysis and detection rules. Secondly, MultiSmell constructs a heterogeneous graph in which features and labels are abstracted into different types of nodes by XLNet, BiLSTM, and CNN models. To generate more expressive node representations, we built a layer to fuse the feature and label nodes by borrowing the message-passing mechanism of graph attention networks. Finally, after several rounds of updating iterations, the label nodes are fed into the multi-layer perceptron to detect multi-label code smells.We evaluate MultiSmell by answering 7 research questions. The experimental results show that our approach exhibits superior exactMatchRatio and F1 scores when detecting multi-label code smells. Furthermore, it improves accuracy by 10.96% and F1 by 8.54% on average compared to the existing approach, demonstrating its effectiveness.


## Data collection
In each iteration of the method-level dataset, the MultiSmell is trained with 27 out of 28 projects and tested on the remaining project samples. The Minitwit project is not included in the testing process as it only contains one kind of code smell and the sample size is too tiny. The process is repeated 27 times.
| Index | Project | Description |
|:------|:------|:------|
| - | Minitwit | Based on Flask's MiniTwit example |
| 1 | ArgoUML-SPL | A project that aims to extract an SPL from the current ArgoUML codebase |
| 2 | Cayenne | An open-source persistence framework, providing object-relational mapping (ORM) and remoting services |
| 3 | Cobertura | A free Java code coverage reporting tool |
| 4 | Displaytag | An open-source suite of custom tags |
| 5 | Fitjava | Integration testing framework for enhanced software development collaboration |
| 6 | Freecs | Open-source test program |
| 7 | Freedomotic | An open-source, flexible, secure Internet of Things (IoT) application framework |
| 8 | HSQLDB | A relational database management system and a set of tools written in Java |
| 9 | iText7 | A powerful PDF toolkit |
| 10 | JAdventure | Java text adventure game |
| 11 | Javacc | The most popular parser generator for use with Java applications |
| 12 | javaStud | Java tutorial example series |
| 13 | JGroups | A clustering library, allowing members to exchange messages |
| 14 | Job | Distributed scheduled job framework |
| 15 | JSmooth | A Java Executable Wrapper that builds standard Windows executable binaries (.exe) that launch java applications |
| 16 | Junit3 | A simple framework to write repeatable tests |
| 17 | Maven | A software project management and comprehension tool |
| 18 | Mylyn | Integration for Eclipse allows you to manage your Redmine issues straight from Eclipse |
| 19 | Nutch | An extensible and scalable web crawler |
| 20 | ParallelColt | A multithreaded version of Colt a library for high performance scientific computing in Java |
| 21 | PMD | A source code analyzer |
| 22 | Rhino | An open-source implementation of JavaScript written entirely in Java |
| 23 | RxJava | A library for composing asynchronous and event-based programs by using observable sequences |
| 24 | SPECjbb2005 | Application Server Testing for Java |
| 25 | Xalan | An XSLT processor for transforming XML documents into HTML, text, or other |
| 26 | Xmlgraphics-batik | A Java-based toolkit for applications that handle images in the Scalable Vector Graphics format for various purposes |
| 27 | Xmojo | JMX specification implementation |

The source part of the dataset is available at the following URL [Data](https://zenodo.org/records/12604914).

## Results of MultiSmell
### RQ1
The detailed assessment metrics values for each project and the confusion matrix.
| Index | Project | hamming_loss | exactMatchRatio | accuracy | precision<sub>macro</sub> | recall<sub>macro</sub> | F1<sub>macro</sub> | precision<sub>micro</sub> | recall<sub>micro</sub> | F1<sub>micro</sub> | confusion matrix |
|:------|:------|:------|:------|:------|:------|:------|:------|:------|:------|:------|:------|
| 1 | ArgoUML-SPL | 0.0125 | 0.9643 | 0.9754 | 0.9884 | 0.9867 | 0.9875 | 0.9885 | 0.9856 | 0.9871 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/1.png) |
| 2 | Cayenne | 0.0052 | 0.9844 | 0.988 | 0.9987 | 0.9893 | 0.9939 | 0.9987 | 0.9892 | 0.9939 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/2.png) |
| 3 | Cobertura | 0.0094 | 0.9722 | 0.9784 | 0.9948 | 0.9834 | 0.989 | 0.9955 | 0.9809 | 0.9881 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/3.png) |
| 4 | Displaytag | 0.002 | 0.9944 | 0.9923 | 1 | 0.9923 | 0.9961 | 1 | 0.9922 | 0.9961 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/4.png) |
| 5 | Fitjava | 0.0218 | 0.9347 | 0.6254 | 0.6506 | 0.6409 | 0.6454 | 0.9753 | 0.9603 | 0.9677 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/5.png) |
| 6 | Freecs | 0.0354 | 0.8973 | 0.9157 | 0.9817 | 0.9292 | 0.9554 | 0.9839 | 0.9311 | 0.9568 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/6.png) |
| 7 | Freedomotic | 0.0126 | 0.9622 | 0.9756 | 0.9856 | 0.9892 | 0.9874 | 0.9856 | 0.9893 | 0.9874 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/7.png) |
| 8 | HSQLDB | 0.0353 | 0.9314 | 0.903 | 0.9544 | 0.9763 | 0.964 | 0.9544 | 0.9754 | 0.9648 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/8.png) |
| 9 | iText7 | 0.0057 | 0.9828 | 0.9888 | 0.9987 | 0.9901 | 0.9944 | 0.9987 | 0.9901 | 0.9944 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/9.png) |
| 10 | JAdventure | 0.0026 | 0.9921 | 0.659 | 0.6667 | 0.659 | 0.6628 | 1 | 0.9884 | 0.9942 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/10.png) |
| 11 | Javacc | 0.0204 | 0.9388 | 0.9452 | 1 | 0.9452 | 0.9712 | 1 | 0.9504 | 0.9746 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/11.png) |
| 12 | javaStud | 0.0032 | 0.9901 | 0.9873 | 1 | 0.9873 | 0.9936 | 1 | 0.987 | 0.9934 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/12.png) |
| 13 | JGroups | 0.0029 | 0.9912 | 0.9918 | 1 | 0.9918 | 0.9959 | 1 | 0.9913 | 0.9956 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/13.png) |
| 14 | Job | 0.0026 | 0.9961 | 0.9898 | 1 | 0.9898 | 0.9949 | 1 | 0.9895 | 0.9947 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/14.png) |
| 15 | JSmooth | 0.009 | 0.9734 | 0.979 | 1 | 0.979 | 0.9894 | 1 | 0.9776 | 0.9887 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/15.png) |
| 16 | Junit3 | 0.0007 | 0.9978 | 0.9986 | 1 | 0.9986 | 0.9993 | 1 | 0.9981 | 0.9991 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/16.png) |
| 17 | Maven | 0.0062 | 0.9814 | 0.9842 | 1 | 0.9841 | 0.992 | 1 | 0.9846 | 0.9922 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/17.png) |
| 18 | Mylyn | 0.0119 | 0.9644 | 0.9733 | 0.9962 | 0.9769 | 0.9864 | 0.9921 | 0.9769 | 0.9864 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/18.png) |
| 19 | Nutch | 0.0125 | 0.005 | 0.9849 | 0.988 | 1 | 0.9939 | 1 | 0.9886 | 0.9943 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/19.png) |
| 20 | ParallelColt | 0.0076 | 0.9772 | 0.9855 | 0.9998 | 0.9857 | 0.9927 | 0.9998 | 0.9845 | 0.9921 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/20.png) |
| 21 | PMD | 0.0021 | 0.9938 | 0.994 | 1 | 0.994 | 0.997 | 1 | 0.9947 | 0.9974 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/21.png) |
| 22 | Rhino | 0.0253 | 0.9313 | 0.9511 | 0.9757 | 0.9752 | 0.9748 | 0.9756 | 0.9746 | 0.9751 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/22.png) |
| 23 | RxJava | 0.0217 | 0.9349 | 0.9496 | 1 | 0.9496 | 0.9736 | 1 | 0.9381 | 0.9681 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/23.png) |
| 24 | SPECjbb2005 | 0.0495 | 0.8594 | 0.9053 | 1 | 0.9053 | 0.9473 | 1 | 0.8827 | 0.9377 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/25.png) |
| 25 | Xalan | 0.0114 | 0.9657 | 0.9772 | 0.9978 | 0.9793 | 0.9883 | 0.9979 | 0.9802 | 0.989 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/25.png) |
| 26 | Xmlgraphics-batik | 0.007 | 0.9791 | 0.9863 | 0.9981 | 0.9881 | 0.9931 | 0.9982 | 0.9882 | 0.9932 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/26.png) |
| 27 | Xmojo | 0.0071 | 0.9788 | 0.9735 | 1 | 0.9735 | 0.9864 | 1 | 0.9725 | 0.986 | ![image text](https://github.com/multismell/MultiSmell/blob/main/confusion_matrix/27.png) |
