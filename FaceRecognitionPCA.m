function FaceRecognitionPCA
clear  % calc xmean,sigma and its eigen decomposition  
close all

%ͼ���ȡ��Ԥ����
xi=[];%����ѵ��ͼ�� 
for i=1:6
    for j=1:4 %ѵ����       
        a=imread(strcat('E:\face\0',num2str(i),num2str(j),'.pgm'));%ͼƬ��ȡ      
        b=a(1:112*92); % b����ʸ�� 1��N������N��10304����ȡ˳�������к��У������ϵ��£�������        
        b=double(b); %��b�����еĲ���ת��Ϊ˫�����Ͳ���      
        xi=[xi; b];  % xi ��һ��M * N ����xi��ÿһ�����ݴ���һ��ͼƬ������M��24 
    end
end

%K_L�㷨���Э�������
psi=mean(xi); % ƽ��ͼƬ��1 �� N  
figure%ƽ��ͼ
imshow(mat2gray(reshape(psi,112,92)));%�ȵõ�һ��112��92����xi�����ľ����ٽ��þ���ת��Ϊ�Ҷ�ͼ�����ʹ�ûҶ�ͼ����ӻ���
for i=1:24
    fai(i,:)=xi(i,:)-psi; % fai��һ��M �� N����faiÿһ�б���������ǡ�ÿ��ͼƬ����-ƽ��ͼƬ�� 
end;   
C=1/24.*fai*fai';   % M * M �׾��� 

%��Э������������������������ֵ�Ĵ�С��������
[lanmuda,d]=eig(C);%lanmuda��d�ֱ�ΪЭ�������C��������һ���������Ͷ�Ӧ������ֵ
d1=diag(d); %d1Ϊd����Խ���Ԫ����ɵľ��󣬶Խ���Ԫ����xi�ķ���
[T,index]=sort(d1); %������������������ֵ�Ӵ�С��˳�����г��У���ɾ���T
cols=size(lanmuda,2);%�����������������
for i=1:cols      
    vsort(:,i) = lanmuda(:, index(cols-i+1) ); % vsort ��һ��M*col(ע:colһ�����M)�׾��󣬱�����ǰ��������е���������,ÿһ�й���һ����������      
    dsort(i)   = d1( index(cols-i+1) );  % dsort �������d1���������е�����ֵ����һά������ 
end  %��ɽ������� 


%����ѡ��90%������(����ֵ��ѡ��) 
dsum = sum(dsort); %�������������ĺ�
dsum_extract = 0;  
p = 0;     
while( dsum_extract/dsum < 0.9)       
    p = p + 1;%����Ҫ��Ĳ��������������ĵ�y/dsum��������Ӧ          
    dsum_extract = sum(dsort(1:p)); %�ѷ���Ҫ���dsort������������ֵ��Ӹ�ֵ��dsum_extract    
end
a=1:1:24;%1��ʾ��ʼԪ�أ�1��ʾ����������24��ʾ��ֹԪ��,a��ʾһά������
for i=1:24
y(i)=sum(dsort(a(1:i)) );%y��1��=dsort��1����y��2��=dsort��1��+dsort��2�����Դ�����
end
figure
y1=ones(1,24);
plot(a,y/dsum,a,y1*0.9,'linewidth',2);%���㼴Ϊ�ܴﵽy1��90%��ǰn������ֵ�ĺ�
grid
title('ǰn��������ռ�ܵ������ٷֱ�');
xlabel('ǰn������ֵ');
ylabel('ռ�ٷֱ�');
figure
plot(a,dsort/dsum,'linewidth',2);
grid
title('��n��������ռ�ܵ������ٷֱ�');
xlabel('��n������ֵ');
ylabel('ռ�ٷֱ�');

%SVD�ֽ�
i=1;  % (ѵ���׶�)�����������γɵ�����ϵ
tic %����ʱ��
while (i<=p && dsort(i)>0)%��i<n�ҵ�i������ֵ>0
    base(:,i) =dsort(i)^(-1/2) * fai' * vsort(:,i);;   % base��N��p�׾��󣬳���dsort(i)^(1/2)�Ƕ�����ͼ��ı�׼����������
      i = i + 1; 
end
toc

%���Լ�ʶ��׶�
allcoor = xi * base;%M��N*N��p=M��p
accu = 0;   % ���Թ��̳�ʼ��
for i=1:6     
    for j=5:8 %����6 x 4 ������ͼ��         
        a=imread(strcat('E:\face\0',num2str(i),num2str(j),'.pgm')); %ͼƬ��ȡ
        b=a(1:112*92);  %n=112*92      
        b=double(b);        
        tcoor= b * base; %�������꣬��1��p�׾���      
        for k=1:24                
            mdist(k)=norm(tcoor-allcoor(k,:));%���ؾ���euclidean����
        end;      

        %���׽���   
        [dist,index2]=sort(mdist);%��mdist����������õ��µ�����dist��index2��ʾ���Ӧ������ɵ�ʸ��         
        class1=floor( index2(1)/4 )+1;%��index2�ĵ�һ����������ÿ���˵Ĳ�����������������������ķ�����ȡ�����ټ�1�õ�����1��class1
        class2=floor(index2(2)/4)+1;        
        class3=floor(index2(3)/4)+1;        
        if class1~=class2 && class2~=class3 
            class=class1; %����ͼ����class1��2��3������ͬһ��        
        elseif class1==class2          
            class=class1;%����ͼ����class1��2����ͬһ�࣬class3����         
        elseif class2==class3     
            class=class2; %����ͼ����class2��3����ͬһ�࣬class1����        
        end;         
        if class==i %����Ĳ���ͼ�������Ĳ���ͼ���Ӧ
            accu=accu+1; %��ȷʶ��ĸ���       
        end;   
    end;
end;  
accu
accuracy=accu/24 %���ʶ����
