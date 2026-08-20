
%% NIR_FINAL_REPRODUCTION
% Reconstruct and test the final NIR workflow using the ORIGINAL split.
%
% Required workspace variables:
%   X, Y
%   and either:
%      trainIndex + testIndex
%   OR:
%      Xtrain + Xtest + Ytrain + Ytest
%
% Optional:
%   VarLabels (rows 4:407 contain the 404 NIR wavenumbers)
%
% The script:
%   1) Uses the original train/test split
%   2) Creates Raw / MSC / SNV / SG1 / SG2 predictors
%   3) Runs PLSR with 10-fold CV (1:20 components)
%   4) Runs SVR with RBF/Gaussian kernel
%   5) Evaluates independent test data
%   6) Saves each model separately
%   7) Saves a complete bundle
%   8) Compares reproduced metrics with dissertation values

clc; close all;

%% -------------------- SETTINGS --------------------
outDir = fullfile(pwd,'NIR_Final_Outputs');
if ~exist(outDir,'dir'), mkdir(outDir); end

sgOrder = 2;
sgWindow = 11;
nCV = 10;
maxLV = 20;

% Reported dissertation values (RMSEP, R2)
target.Raw.PLS = [0.3788, 0.9169];
target.Raw.SVR = [0.3283, 0.9376];
target.MSC.PLS = [0.3658, 0.9225];
target.MSC.SVR = [0.3068, 0.9455];
target.SNV.PLS = [0.3655, 0.9226];
target.SNV.SVR = [0.2284, 0.9698];
target.SG1.PLS = [0.3620, 0.9241];
target.SG1.SVR = [0.3007, 0.9476];
target.SG2.PLS = [0.3815, 0.9157];
target.SG2.SVR = [0.3844, 0.9144];

%% -------------------- CHECK WORKSPACE --------------------
assert(exist('X','var')==1,'X is missing.');
assert(exist('Y','var')==1,'Y is missing.');

X = double(X);
Y = double(Y(:));

assert(size(X,1)==numel(Y),'X and Y sample counts do not match.');
assert(size(X,2)==404,'Expected NIR X to contain 404 spectral variables.');

%% -------------------- ORIGINAL SPLIT --------------------
[trainIndex,testIndex,Xtrain0,Xtest0,Ytrain,Ytest] = resolveOriginalSplit( ...
    X,Y,'trainIndex','testIndex');

fprintf('Original split: %d train / %d test\n',numel(trainIndex),numel(testIndex));

%% -------------------- WAVENUMBER --------------------
wavenumber = [];
if exist('VarLabels','var')
    try
        wavenumber = str2double(strtrim(cellstr(VarLabels(4:407,:))));
        wavenumber = wavenumber(:);
        if numel(wavenumber)~=size(X,2), wavenumber=[]; end
    catch
        wavenumber=[];
    end
end

%% -------------------- PREPROCESSING --------------------
Xprep.Raw = X;
Xprep.MSC = applyMSC(X);
Xprep.SNV = applySNV(X);
Xprep.SG1 = applySGDerivative(X,sgOrder,sgWindow,1);
Xprep.SG2 = applySGDerivative(X,sgOrder,sgWindow,2);

%% -------------------- OPTIONAL SPECTRUM FIGURES --------------------
if ~isempty(wavenumber)
    makeNIRFigureSet(Xprep,wavenumber,outDir);
end

%% -------------------- MODEL LOOP --------------------
prepNames = fieldnames(Xprep);
rows = [];
Models = struct();

for p = 1:numel(prepNames)
    prep = prepNames{p};
    Xi = Xprep.(prep);

    Xtr = Xi(trainIndex,:);
    Xte = Xi(testIndex,:);

    % ---------- PLSR ----------
    [B,PLSstruct,cvRMSE,lvSelected] = fitPLSFinal(Xtr,Ytrain,nCV,maxLV);
    YpredPLS = [ones(size(Xte,1),1), Xte] * B;
    [rmsepPLS,r2PLS,resPLS] = metrics(Ytest,YpredPLS);

    Models.(prep).PLS.B = B;
    Models.(prep).PLS.Struct = PLSstruct;
    Models.(prep).PLS.CVRMSE = cvRMSE;
    Models.(prep).PLS.NumComponents = lvSelected;
    Models.(prep).PLS.Ypred = YpredPLS;
    Models.(prep).PLS.Residuals = resPLS;
    Models.(prep).PLS.RMSEP = rmsepPLS;
    Models.(prep).PLS.R2 = r2PLS;

    save(fullfile(outDir,sprintf('NIR_%s_PLSR_Model.mat',prep)), ...
        'B','PLSstruct','cvRMSE','lvSelected','YpredPLS','resPLS','rmsepPLS','r2PLS', ...
        'trainIndex','testIndex','Ytrain','Ytest');

    rows = [rows; {prep,'PLS',rmsepPLS,r2PLS,lvSelected}];

    % ---------- SVR ----------
    % Final dissertation convention:
    % NIR SVR: automatic/default hyperparameters.
    SVRModel = fitrsvm(Xtr,Ytrain,'KernelFunction','gaussian', ...
        'Standardize',true,'OptimizeHyperparameters','auto', ...
        'HyperparameterOptimizationOptions',struct( ...
            'ShowPlots',false,'Verbose',0,'UseParallel',false));

    YpredSVR = predict(SVRModel,Xte);
    [rmsepSVR,r2SVR,resSVR] = metrics(Ytest,YpredSVR);

    Models.(prep).SVR.Model = SVRModel;
    Models.(prep).SVR.Ypred = YpredSVR;
    Models.(prep).SVR.Residuals = resSVR;
    Models.(prep).SVR.RMSEP = rmsepSVR;
    Models.(prep).SVR.R2 = r2SVR;

    save(fullfile(outDir,sprintf('NIR_%s_SVR_Model.mat',prep)), ...
        'SVRModel','YpredSVR','resSVR','rmsepSVR','r2SVR', ...
        'trainIndex','testIndex','Ytrain','Ytest');

    rows = [rows; {prep,'SVR',rmsepSVR,r2SVR,NaN}];

    fprintf('%-4s %-3s RMSEP=%0.6f  R2=%0.6f\n',prep,'PLS',rmsepPLS,r2PLS);
    fprintf('%-4s %-3s RMSEP=%0.6f  R2=%0.6f\n',prep,'SVR',rmsepSVR,r2SVR);
end

Results = cell2table(rows,'VariableNames',{'Preprocessing','Model','RMSEP','R2','NumComponents'});

%% -------------------- REPRODUCTION CHECK --------------------
for i = 1:height(Results)
    prep = Results.Preprocessing{i};
    model = Results.Model{i};
    calc = [Results.RMSEP(i),Results.R2(i)];
    ref = target.(prep).(model);
    Results.AbsDiffRMSEP(i) = abs(calc(1)-ref(1));
    Results.AbsDiffR2(i) = abs(calc(2)-ref(2));
end

writetable(Results,fullfile(outDir,'NIR_Model_Performance_Reproduction.csv'));

save(fullfile(outDir,'NIR_Final_Model_Bundle.mat'), ...
    'Models','Results','Xprep','wavenumber','trainIndex','testIndex','Ytrain','Ytest', ...
    'sgOrder','sgWindow','nCV','maxLV','target');

disp(Results);

%% -------------------- SELECTED MODEL SAVES --------------------
% These are the models highlighted in the dissertation.
Selected_NIR_SNV_SVR = Models.SNV.SVR;
Selected_NIR_SG1_PLS  = Models.SG1.PLS;
save(fullfile(outDir,'NIR_Selected_Models.mat'), ...
    'Selected_NIR_SNV_SVR','Selected_NIR_SG1_PLS');

fprintf('\nNIR reproduction complete. Outputs saved to:\n%s\n',outDir);

%% ==================== LOCAL FUNCTIONS ====================

function [trainIndex,testIndex,Xtr,Xte,Ytr,Yte] = resolveOriginalSplit(X,Y,trName,teName)
    if evalin('base',sprintf('exist(''%s'',''var'')',trName)) && ...
       evalin('base',sprintf('exist(''%s'',''var'')',teName))
        trainIndex = evalin('base',trName);
        testIndex  = evalin('base',teName);
        trainIndex = trainIndex(:);
        testIndex  = testIndex(:);
    elseif evalin('base','exist(''Xtrain'',''var'')') && ...
           evalin('base','exist(''Xtest'',''var'')') && ...
           evalin('base','exist(''Ytrain'',''var'')') && ...
           evalin('base','exist(''Ytest'',''var'')')
        Xtr = evalin('base','Xtrain');
        Xte = evalin('base','Xtest');
        Ytr = evalin('base','Ytrain'); Ytr=Ytr(:);
        Yte = evalin('base','Ytest');  Yte=Yte(:);
        return;
    else
        error(['Original train/test split not found. Provide trainIndex/testIndex ' ...
               'or Xtrain/Xtest/Ytrain/Ytest.']);
    end
    Xtr = X(trainIndex,:); Xte = X(testIndex,:);
    Ytr = Y(trainIndex); Yte = Y(testIndex);
end

function Xmsc = applyMSC(X)
    ref = mean(X,1);
    Xmsc = zeros(size(X));
    for i=1:size(X,1)
        p = polyfit(ref,X(i,:),1);
        if abs(p(1))<eps, error('MSC slope is zero at sample %d.',i); end
        Xmsc(i,:)=(X(i,:)-p(2))./p(1);
    end
end

function Xsnv = applySNV(X)
    Xsnv = zeros(size(X));
    for i=1:size(X,1)
        mu=mean(X(i,:)); s=std(X(i,:),0,2);
        if s<eps, error('SNV standard deviation is zero at sample %d.',i); end
        Xsnv(i,:)=(X(i,:)-mu)./s;
    end
end

function Xd = applySGDerivative(X,order,frame,derivOrder)
    [~,G]=sgolay(order,frame);
    Xd=zeros(size(X));
    for i=1:size(X,1)
        x=X(i,:)';
        % Coefficients used as reconstructed in the final workflow.
        Xd(i,:)=conv(x,G(:,derivOrder+1),'same')';
    end
end

function [B,mdl,cvRMSE,lv] = fitPLSFinal(Xtr,Ytr,nCV,maxLV)
    maxLV=min([maxLV,size(Xtr,1)-1,size(Xtr,2)]);
    cv=crossvalind('Kfold',size(Xtr,1),nCV);
    mse=zeros(maxLV,1);
    for a=1:maxLV
        pred=zeros(size(Ytr));
        for k=1:nCV
            tr=cv~=k; va=cv==k;
            [~,~,~,~,b]=plsregress(Xtr(tr,:),Ytr(tr),a);
            pred(va)= [ones(sum(va),1),Xtr(va,:)]*b;
        end
        mse(a)=mean((Ytr-pred).^2);
    end
    [~,lv]=min(mse);
    [~,~,~,~,B]=plsregress(Xtr,Ytr,lv);
    mdl = struct('NumComponents',lv);
    cvRMSE=sqrt(mse);
end

function [rmsep,r2,res] = metrics(y,yhat)
    y=y(:); yhat=yhat(:);
    res=y-yhat;
    rmsep=sqrt(mean(res.^2));
    sse=sum((y-yhat).^2);
    sst=sum((y-mean(y)).^2);
    r2=1-sse/sst;
end

function makeNIRFigureSet(Xp,wv,outDir)
    names=fieldnames(Xp);
    for i=1:numel(names)
        prep=names{i};
        f=figure('Color','w','Visible','off');
        plot(wv,Xp.(prep)','LineWidth',0.7);
        xlabel('Wavenumber (cm^{-1})'); ylabel('Absorbance / Transformed Value');
        title(['NIR ',prep,' Spectra']);
        grid on; box on; xlim([min(wv) max(wv)]);
        exportgraphics(f,fullfile(outDir,['NIR_',prep,'_Spectra.png']),'Resolution',300);
        close(f);
    end
end
