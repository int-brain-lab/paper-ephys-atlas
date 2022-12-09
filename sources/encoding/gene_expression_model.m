%%%%%%%%%% Gene_model(66,57,40,114) %%%%%%%%%%%%%

%%%%%%%%%% Allen_gene_model(275,58,41,67) %%%%%%%




%%%%%%%%%%%%% Table of voltage features = (number of unit, number of features) %%%%%%%%%%%%%%%

%Y(spike_rate,psd_delta,psd_theta,psd_alpha,psd_beta,psd_gamma,rms_ap,rms_if)


Y=[chann_rate(:,4),chann_volt(:,4:10)];
Y_index=chann_rate(:,1:3);

num_unit=size(Y,1);
num_gene=size(Reduced_allen_gene,1);


%%%%%%%%%%%%%%%%% model: Reduced_allen_gene= (number of PC:275, x,y,z) %%%%%%%%%%%%
  
X_model=zeros(num_unit,num_gene);


for i_unit=1:num_unit
    
    
    test_y=max(ceil(Y_index(i_unit,1)/8),1);
    test_x=max(ceil(Y_index(i_unit,2)/8),1);
    test_z=max(ceil(Y_index(i_unit,3)/8),1);
    
 
    X_model(i_unit,:)=Reduced_allen_gene(:,test_y,test_z,test_x);
    
 
end

id=find(isnan(X_model(:,2))| isnan(Y(:,1)));

X_model(id,:)=[];
Y(id,:)=[];

Y(find(isnan(Y(:,7)) ),7)=0;

%%%%%%%%%%%%% ridge-regression model %%%%%%%%%%%%%%%%%%


% Y=  X * b;
% Y * beta_inv= X;
% Y =  X* (beta_inv)^-1;

%[L_model, beta_model] = ridgeMML(Y,X, true);
[L_model, beta_model] = ridgeMML(Y,X_model, 1);


 Ym= (X_model - mean(X_model, 1)) * beta_model;
%Ym= X * beta_model(2:end,:)+beta_model(1,:);




%%%%%%%%%%%%%%%%%% R^2  %%%%%%%%%%%%%%%%%%

Rsq_allen_model=zeros(size(Y,2),1);

for sdi=1:size(Y,2)


y1=reshape(Y(:,sdi),[],1);
y1m=reshape(Ym(:,sdi),[],1);

%Rsq_pixel(sdi,1) = 1-sum( (y1-y1m).^2 )/sum((y1 - mean(y1)).^2);


Rsq_allen_model(sdi,1) = corr(y1,y1m)^2;


end

save('Rsq_allen_gene_model.mat','Rsq_allen_model')


%%%%%%%%%%%%%% CV R^2  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[Ym21,beta1,cRidge1,errorbar] =  crossValModel(Y,X_model,10);
CV_model=zeros(size(Y,2),11);

%%% Rsq= (sptaial dim of video, 1);


for sdi=1:size(Y,2)



y1=reshape(Y(:,sdi),[],1);
y1m=reshape(Ym21(:,sdi),[],1);


CV_model(sdi,1)= corr(y1,y1m)^2;
CV_model(sdi,2:11)= (errorbar(sdi,:));

end




%%%%%%% delta R^2 for each gene  %%%%%%%%%%%%%%%%%%

R_2=zeros(size(Y,2),size(X_model,2));
delta_R=zeros(size(Y,2),size(X_model,2));
%%% (num_voltage_feature, num_gene) %%%%

for i_gene=1:size(X_model,2)

X_perm=X_model;
X_perm(:,i_gene)=X_model(randperm(size(X_perm,1)),i_gene);


[Ym21,beta1,cRidge1,errorbar] =  crossValModel(Y,X_perm,10);

R_2(:,i_gene)=nanmean(errorbar,2);

delta_R(:,i_gene)=nanmean(CV_model,2)-R_2(:,i_gene);

end
