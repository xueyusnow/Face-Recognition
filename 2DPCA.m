function Face
clear  
close all

%ͼ���ȡ��Ԥ����
xi=[];%����ѵ��ͼ�� 
M=1;
for i=1:6
    for j=1:4 %ѵ����       
       a=imread(strcat('E:\face\0',num2str(i),num2str(j),'.pgm'));%ͼƬ��ȡ 
       A(:,:,M)=a;
       A=double(A);
       M=M+1;
    end
end

%��������ͶӰ����
sum=zeros(size(A(:,:,1)));
for t=1:M-1 %ѵ����       
       sum=sum+A(:,:,t);
end
psi=1/(M-1)*sum; %ƽ��ֵ
figure %ƽ��ͼ
imshow(mat2gray(reshape(psi,112,92)));
for i=1:24
fai(:,:,i)=A(:,:,i)-psi; 
end;   
sum3=zeros(size(fai(:,:,1)*fai(:,:,1)'));
for i=1:24
    sum3=sum3+fai(:,:,i)*fai(:,:,i)'; 
end;
Gl=1/24*sum3; %����ɢ������
[v,d]=eig(Gl);
[d_sort,index3]=dsort(diag(d));
F=v(:,index3);
P = F(:, 1:50); %����ͶӰ����

%������ȡ������ʱ��
tic
for j=1:24
B(:,:,j) = A(:,:,j)'*P; 
end
toc

%��С���������
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
    for j=5:8 %ѵ����       
       a=imread(strcat('E:\face\0',num2str(i),num2str(j),'.pgm'));%ͼƬ��ȡ 
       A(:,:,M)=a;
       A=double(A);
       BBB=A(:,:,M)'*P;
       for n=1:6
           HH(n)=norm(BB(:,:,n)-BBB); %������euclidean����
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
accuracy=accu/24 %���ʶ����
