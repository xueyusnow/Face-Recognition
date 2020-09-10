function Face_M2DPCA
clear  % calc xmean,sigma and its eigen decomposition  
close all
xi=[];%所有训练图像 

%图像读取

M=1;
for i=1:8
    for j=1:4 %训练集             
        a=imread(strcat('E:\face\0',num2str(i),num2str(j),'.pgm'));%图片读取
        a=double(a);
        MA=mat2cell(a, [56,56], [23,23,23,23]);
        A(:,:,M)=a;
        for p=1:2
            for q=1:4
                MMA(8*(M-1)+4*(p-1)+q)=MA(p,q);
%        MMA(p,q)=cell2mat(MA(p,q));
%                  FFA=MMA(4*(p-1)+q);
%             sum1(p,1)=sum1(p,1)+FFA(p,q);
            end
        end
        M=M+1;
    end
end
% m1=cell2mat(MMA(1))
% m2=cell2mat(MMA(2))
% A(:,:,1)
% A(:,:,1)'
% sum1=zeros(size(MA));
% sum=zeros(size(MA));

SUMA=zeros(size(56,23));
i=1;
     for t=1:32
         for j=1:4
             FAA=MMA(8*(t-1)+j);
             MMMA=cell2mat(FAA);
%              size(MMMA)
             SUMA=SUMA+MMMA;
%         sum1(i,:)=sum1(i,:)+MMMA(8*(t-1)+4*(i-1)+j);
         end
        
     end
     SUMB(1:56,1:23)=SUMA;
% SUMA=zeros(size(56,23));
i=2;
     for t=1:32
         for j=1:4
             FAA=MMA(8*(t-1)+4*(i-1)+j);
             MMMA=cell2mat(FAA);
%              size(MMMA)
             SUMA=SUMA+MMMA;
%         sum1(i,:)=sum1(i,:)+MMMA(16*(t-1)+4*(i-1)+j);
         end  
     end
%      SUMB(57:112,1:23)=SUMA;

 B=SUMA/32/2/4;
 
 SUM=zeros(size(56,56));
i=1;
     for t=1:32
         for j=1:4
             FAA=MMA(8*(t-1)+j);
             MMMA=cell2mat(FAA);
%              size(MMMA)
             SUM=SUM+(MMMA-B)*(MMMA-B)';
%              size(SUM)
%         sum1(i,:)=sum1(i,:)+MMMA(8*(t-1)+4*(i-1)+j);
         end
        
     end
i=2;
     for t=1:32
         for j=1:4
             FAA=MMA(8*(t-1)+4*(i-1)+j);
             MMMA=cell2mat(FAA);
%              size(MMMA)
             SUM=SUM+(MMMA-B)*(MMMA-B)';
%         sum1(i,:)=sum1(i,:)+MMMA(16*(t-1)+4*(i-1)+j);
         end  
     end
G2=SUM/32/2/4;
size(G2)
[v,d]=eig(G2);
[d_sort,index3]=dsort(diag(d));
F=v(:,index3);
% size(F)
d=5;
P = F(:, 1:d);
% size(P)

%特征提取

tic
% for M=1:32
    for p=1:2
        for q=1:4
            FAA=MMA(4*(p-1)+q);
            MMMA=cell2mat(FAA);
%         size(MMMA)
            Bi(23*(p-1)+1:p*23,d*(q-1)+1:q*d)=MMMA'*P;
        end
    end
% FB=Bi;
% size(Bi)
% B(1:46,(M-1)*4*d+1:M*4*d)=Bi;
% B(:,:,M)=Bi
% end
toc

%分类
% n=0;
for t=1:8 
    sum2=zeros(size(A(:,:,1)));
    for m=1:4
        g=(t-1)*4+m;
       sum2=sum2+A(:,:,g);
    end
    AA(:,:,t)=1/4*sum2;
%     size(AA(:,:,t))
    MEANA=mat2cell(AA(:,:,t), [56,56], [23,23,23,23]);
    for p=1:2
    for q=1:4
        MEAN=cell2mat(MEANA(p,q));
        XEANBi(23*(p-1)+1:p*23,d*(q-1)+1:q*d)=MEAN'*P;
%         n=n+1
    end
    end
    XEANB(:,:,t)=XEANBi;
end
% for M=1:24

accu = 0; 
M=1;
for i=1:8
    for j=5:8 %训练集       
       a=imread(strcat('E:\face\0',num2str(i),num2str(j),'.pgm'));%图片读取 
       a=double(a);
%        B(:,:,M)=a;
       MB=mat2cell(a, [56,56], [23,23,23,23]);
       for p=1:2
        for q=1:4
            FBB=MB(p,q);
            MMMB=cell2mat(FBB);
%         size(MMMA)
            BBB(23*(p-1)+1:p*23,d*(q-1)+1:q*d)=MMMB'*P;
        end
       end
       for n=1:8
           HH(n)=norm(XEANB(:,:,n)-BBB);
       end 
       [HHH,index2]=sort(HH);
       H=HHH(:,1);   
       Bl=norm(XEANB(:,:,i)-BBB);
       if H==Bl
          accu=accu+1;
       end
      M=M+1;
    end
    
end
accu
accuracy=accu/32 %输出识别率