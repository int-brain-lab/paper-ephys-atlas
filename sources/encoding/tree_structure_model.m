%%%%%%%%%%%%% table of voltage feature %%%%%%%%%%%%%%%


%Y(spike_rate,psd_delta,psd_theta,psd_alpha,psd_beta,psd_gamma,rms_ap,rms_if)


Y=[chann_rate(:,4),chann_volt(:,4:10)];
Y_index=chann_rate(:,1:3);

num_unit=size(Y,1);
num_structure=size(tree_atlas,1);


%%%%%%%%%%%%% tree-structure (number of region, x,y,z) %%%%%%%%%%%%%%%


X_model=zeros(num_unit,num_structure);


for i_unit=1:num_unit
    
    
    test_y=max(ceil(Y_index(i_unit,1)/8),1);
    test_x=max(ceil(Y_index(i_unit,2)/8),1);
    test_z=max(ceil(Y_index(i_unit,3)/8),1);
    
    X_model(i_unit,:)=tree_atlas(:,test_x,test_y,test_z);
    
 
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

Rsq_tree_model=zeros(size(Y,2),1);

for sdi=1:size(Y,2)


y1=reshape(Y(:,sdi),[],1);
y1m=reshape(Ym(:,sdi),[],1);

%Rsq_pixel(sdi,1) = 1-sum( (y1-y1m).^2 )/sum((y1 - mean(y1)).^2);


Rsq_tree_model(sdi,1) = corr(y1,y1m)^2;


end

%save('Rsq_tree_model.mat','Rsq_tree_model')


%%%%%%%%%%%%%% CV R^2  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[Ym21,beta1,cRidge1,errorbar] =  crossValModel(Y,X_model,10,0);
CV_model=zeros(size(Y,2),11);

%%% Rsq= (sptaial dim of video, 1);


for sdi=1:size(Y,2)



y1=reshape(Y(:,sdi),[],1);
y1m=reshape(Ym21(:,sdi),[],1);


CV_model(sdi,1)= corr(y1,y1m)^2;
CV_model(sdi,2:11)= (errorbar(sdi,:));

end

mean_CV_model=nanmean(CV_model(:,2:11),2);




save('Rsq_tree_model.mat','Rsq_tree_model','CV_model','mean_CV_model')



%%%%%%%%%%%%% Tree+Gene model %%%%%%%%%%%%%%%%%%%%%%%%%%



%% select top 275 PC of gene-expression atlas %%
num_pc=275;



Y=[chann_rate(:,4),chann_volt(:,4:10)];
Y_index=chann_rate(:,1:3);

z=reshape(allen_gene,size(allen_gene,1),size(allen_gene,2)*size(allen_gene,3)*size(allen_gene,4));

W_1=W(:,1:num_pc);


Reduced_allen_gene_0=z'*W_1;

Reduced_allen_gene=reshape(Reduced_allen_gene_0,size(allen_gene,2),size(allen_gene,3),size(allen_gene,4),size(Reduced_allen_gene_0,2));



num_gene=size(Reduced_allen_gene,4);
num_structure=size(tree_atlas,1);



X_model=zeros(num_unit,num_gene+num_structure);


for i_unit=1:num_unit
    
    
    test_y=max(ceil(Y_index(i_unit,1)/8),1);
    test_x=max(ceil(Y_index(i_unit,2)/8),1);
    test_z=max(ceil(Y_index(i_unit,3)/8),1);
    
   % X_model(i_unit,:)=Gene_model_sym(test_x,test_y,test_z,:);
    X_model(i_unit,1:num_gene)=Reduced_allen_gene(test_y,test_z,test_x,:);
    X_model(i_unit,num_gene+1:end)=tree_atlas(:,test_x,test_y,test_z)';
    
 
end

id=find(isnan(X_model(:,2))| isnan(Y(:,1)));

X_model(id,:)=[];
Y(id,:)=[];

Y(find(isnan(Y(:,7)) ),7)=0;


%%%%%%%%%%%%%% CV R^2  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[Ym21,beta1,cRidge1,errorbar] =  crossValModel(Y,X_model,10,0);
CV_model=zeros(size(Y,2),11);

%%% Rsq= (sptaial dim of video, 1);


for sdi=1:size(Y,2)



y1=reshape(Y(:,sdi),[],1);
y1m=reshape(Ym21(:,sdi),[],1);


CV_model(sdi,1)= corr(y1,y1m)^2;
CV_model(sdi,2:11)= (errorbar(sdi,:));

end

mean_CV_model_gene_and_tree=nanmean(CV_model(:,2:11),2);
