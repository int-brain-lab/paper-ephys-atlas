function [Vm, cBeta, cRidge,errorbar] =  crossValModel(Vc,fullR,folds,rand_crit)
% function to compute cross-validated R^2




cR = fullR;

Vm = zeros(size(Vc),'single'); %pre-allocate motor-reconstructed V
rng(1) % for reproducibility
if rand_crit==1
   randIdx = randperm(size(Vc,1)); %generate randum number index
elseif rand_crit==0
   randIdx = 1:1:size(Vc,1);
end
    
foldCnt = floor(size(Vc,1) / folds);
cBeta = cell(1,folds);


errorbar=zeros(size(Vc,2),folds);

for iFolds = 1:folds
    dataIdx = true(1,size(Vc,1));
    
    if folds > 1
        dataIdx(randIdx(((iFolds - 1)*foldCnt) + (1:foldCnt))) = false; %index for training data
        if iFolds == 1
            [cRidge, cBeta{iFolds}] = ridgeMML(Vc(dataIdx,:), cR(dataIdx,:), 1); %get beta weights and ridge penalty for task only model
        else
            [~, cBeta{iFolds}] = ridgeMML(Vc(dataIdx,:), cR(dataIdx,:), 1, cRidge); %get beta weights for task only model. ridge value should be the same as in the first run.
        end
        
        Vm(~dataIdx,:) = ((cR(~dataIdx,:)-nanmean(cR(~dataIdx,:),1)) * cBeta{iFolds}); %predict remaining data
        
        if rem(iFolds,folds/5) == 0
            fprintf(1, 'Current fold is %d out of %d\n', iFolds, folds);
        end
    else
        [cRidge, cBeta{iFolds}] = ridgeMML(Vc, cR, 1); %get beta weights for task-only model.
        Vm = ((cR-nanmean(cR,1)) * cBeta{iFolds}); %predict remaining data
        disp('Ridgefold is <= 1, fit to complete dataset instead');
    end
    
    
    Y1=Vc(~dataIdx,:);
    Y1m=Vm(~dataIdx,:);
    
    for sdi=1:size(Vc,2)
    
   y1=Y1(:,sdi);
   y1m=Y1m(:,sdi);   
    
   
   errorbar(sdi,iFolds)=corr(y1,y1m)^2;
        
        
    end
    
    
  
    
    
end



end