function Face
clear  
close all

%图像读取和预处理
xi=[];%所有训练图像 
M=1;
for i=1:6
    for j=1:4 %训练集       
       a=imread(strcat('E:\face\0',num2str(i),num2str(j),'.pgm'));%图片读取 
       A(:,:,M)=a;
       A=double(A);
       M=M+1;
    end
end

%计算最优投影向量
sum=zeros(size(A(:,:,1)));
for t=1:M-1 %训练集       
       sum=sum+A(:,:,t);
end
psi=1/(M-1)*sum; %平均值
figure %平均图
imshow(mat2gray(reshape(psi,112,92)));
for i=1:24
fai(:,:,i)=A(:,:,i)-psi; 
end;   
sum3=zeros(size(fai(:,:,1)*fai(:,:,1)'));
for i=1:24
    sum3=sum3+fai(:,:,i)*fai(:,:,i)'; 
end;
Gl=1/24*sum3; %总体散布矩阵
[v,d]=eig(Gl);
[d_sort,index3]=dsort(diag(d));
F=v(:,index3);
P = F(:, 1:50); %最优投影矩阵

%特征抽取并计算时间
tic
for j=1:24
B(:,:,j) = A(:,:,j)'*P; 
end
toc

%最小距离分类器
for t=1:6 
    sum2=zeros(size(A(:,:,1)));
    for m=1:4
        g=(t-1)*4+m;
       sum2=sum2+A(:,:,g);
    end
    AA(:,:,t)=1/4*sum2;
    BB(:,:,t) = AA(:,:,t)' *P;
end
accu = 0; 
M=25;
for i=1:6
    for j=5:8 %训练集       
       a=imread(strcat('E:\face\0',num2str(i),num2str(j),'.pgm'));%图片读取 
       A(:,:,M)=a;
       A=double(A);
       BBB=A(:,:,M)'*P;
       for n=1:6
           HH(n)=norm(BB(:,:,n)-BBB); %求矩阵的euclidean距离
       end 
       [HHH,index2]=sort(HH);
       H=HHH(:,1);   
       Bl=norm(BB(:,:,i)-BBB);
       if H==Bl
          accu=accu+1;
       end
      M=M+1;
    end
    
end
accu
accuracy=accu/24 %输出识别率
