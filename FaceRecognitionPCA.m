function FaceRecognitionPCA
clear  % calc xmean,sigma and its eigen decomposition  
close all

%图像读取和预处理
xi=[];%所有训练图像 
for i=1:6
    for j=1:4 %训练集       
        a=imread(strcat('E:\face\0',num2str(i),num2str(j),'.pgm'));%图片读取      
        b=a(1:112*92); % b是行矢量 1×N，其中N＝10304，提取顺序是先列后行，即从上到下，从左到右        
        b=double(b); %将b矩阵中的参数转化为双精度型参数      
        xi=[xi; b];  % xi 是一个M * N 矩阵，xi中每一行数据代表一张图片，其中M＝24 
    end
end

%K_L算法求得协方差矩阵
psi=mean(xi); % 平均图片，1 × N  
figure%平均图
imshow(mat2gray(reshape(psi,112,92)));%先得到一个112×92的由xi而来的矩阵，再将该矩阵转化为灰度图像，最后使该灰度图像可视化。
for i=1:24
    fai(i,:)=xi(i,:)-psi; % fai是一个M × N矩阵，fai每一行保存的数据是“每个图片数据-平均图片” 
end;   
C=1/24.*fai*fai';   % M * M 阶矩阵 

%将协方差矩阵的特征向量按照特征值的大小降序排列
[lanmuda,d]=eig(C);%lanmuda和d分别为协方差矩阵C的正交归一特征向量和对应的特征值
d1=diag(d); %d1为d矩阵对角线元素组成的矩阵，对角线元素是xi的方差
[T,index]=sort(d1); %将特征向量按照特征值从大到小的顺序排列成行，组成矩阵T
cols=size(lanmuda,2);%特征向量矩阵的列数
for i=1:cols      
    vsort(:,i) = lanmuda(:, index(cols-i+1) ); % vsort 是一个M*col(注:col一般等于M)阶矩阵，保存的是按降序排列的特征向量,每一列构成一个特征向量      
    dsort(i)   = d1( index(cols-i+1) );  % dsort 保存的是d1按降序排列的特征值，是一维行向量 
end  %完成降序排列 


%以下选择90%的能量(特征值的选择) 
dsum = sum(dsort); %求行向量参数的和
dsum_extract = 0;  
p = 0;     
while( dsum_extract/dsum < 0.9)       
    p = p + 1;%符合要求的参数的数量与下文的y/dsum的数量对应          
    dsum_extract = sum(dsort(1:p)); %把符合要求的dsort的行向量参数值相加赋值给dsum_extract    
end
a=1:1:24;%1表示起始元素，1表示递增个数，24表示终止元素,a表示一维行向量
for i=1:24
y(i)=sum(dsort(a(1:i)) );%y（1）=dsort（1），y（2）=dsort（1）+dsort（2），以此类推
end
figure
y1=ones(1,24);
plot(a,y/dsum,a,y1*0.9,'linewidth',2);%交点即为能达到y1的90%的前n个特征值的和
grid
title('前n个特征特占总的能量百分比');
xlabel('前n个特征值');
ylabel('占百分比');
figure
plot(a,dsort/dsum,'linewidth',2);
grid
title('第n个特征特占总的能量百分比');
xlabel('第n个特征值');
ylabel('占百分比');

%SVD分解
i=1;  % (训练阶段)计算特征脸形成的坐标系
tic %计算时间
while (i<=p && dsort(i)>0)%当i<n且第i个特征值>0
    base(:,i) =dsort(i)^(-1/2) * fai' * vsort(:,i);;   % base是N×p阶矩阵，除以dsort(i)^(1/2)是对人脸图像的标准化，特征脸
      i = i + 1; 
end
toc

%测试集识别阶段
allcoor = xi * base;%M×N*N×p=M×p
accu = 0;   % 测试过程初始化
for i=1:6     
    for j=5:8 %读入6 x 4 副测试图像         
        a=imread(strcat('E:\face\0',num2str(i),num2str(j),'.pgm')); %图片读取
        b=a(1:112*92);  %n=112*92      
        b=double(b);        
        tcoor= b * base; %计算坐标，是1×p阶矩阵      
        for k=1:24                
            mdist(k)=norm(tcoor-allcoor(k,:));%返回矩阵euclidean距离
        end;      

        %三阶近邻   
        [dist,index2]=sort(mdist);%将mdist以升序排序得到新的向量dist，index2表示其对应次序组成的矢量         
        class1=floor( index2(1)/4 )+1;%将index2的第一个参数除以每个人的测试样本数量，再往负无穷的方向上取整，再加1得到类型1即class1
        class2=floor(index2(2)/4)+1;        
        class3=floor(index2(3)/4)+1;        
        if class1~=class2 && class2~=class3 
            class=class1; %测试图像与class1、2、3都属于同一类        
        elseif class1==class2          
            class=class1;%测试图像与class1、2属于同一类，class3不是         
        elseif class2==class3     
            class=class2; %测试图像与class2、3属于同一类，class1不是        
        end;         
        if class==i %解算的测试图像与参与的测试图像对应
            accu=accu+1; %正确识别的个数       
        end;   
    end;
end;  
accu
accuracy=accu/24 %输出识别率
