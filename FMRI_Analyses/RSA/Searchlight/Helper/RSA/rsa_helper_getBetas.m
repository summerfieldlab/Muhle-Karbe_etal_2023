function [b,events] = rsa_helper_getBetas(SPM,nRuns,nConds,nMotRegs,maskIndices)
  %% [b,events] = rsa_helper_getBetas(SPM)
  %
  % imports betas from nifti files of single subject
  % returns them together with event IDs (from glm)
  %
  % Timo Flesch, 2019
  % Human Information Processing Lab
  % University of Oxford


  % discard motion regressors and constant terms (for run),
  % assume that each run was modeled with separate set of regressors:
  nTotal      = length(SPM.Vbeta);
  betaCounts  =          1:nTotal;
  betaMask    = [repmat([ones(1,nConds),zeros(1,nMotRegs)],1,nRuns),zeros(1,nRuns)];
  betaIDs     =  find(betaMask==1);
  betasToLoad = SPM.Vbeta(betaIDs);
%   for i = 1:length(betasToLoad)
%     betasToLoad(i).fname = strcat('r',betasToLoad(i).fname);
%     betasToLoad(i).dim   = [66 78 66];
%     betasToLoad(i).mat   = [-3 0 0 -99.5; 0 3 0 -135; 0 0 3 -81; 0 0 0 1];
%   end
  if exist('maskIndices','var')
    b = spm_data_read(betasToLoad, maskIndices);
  else
    b = spm_data_read(betasToLoad);
  end
  events =                    {betasToLoad.descrip};

end