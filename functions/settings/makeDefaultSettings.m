%% Write Retrospective json parameter file
%
% Gustav Strijkers
% Amsterdam UMC
% g.j.strijkers@amsterdamumc.nl
% 23 July 2025
%
%

clearvars;
clc;



%% Define jsonPars


% Versionchecks

jsonPar.newestVersions = {"10.5","10.4","10.3","10.2","10.1","10.0","9.9","9.8","9.7","9.6","9.5","9.4"};
jsonPar.searchStr = "retrospective_version=";
jsonPar.web1 = "https://github.com/Moby1971/Retrospective-v";
jsonPar.web2 = "/blob/main/appversion.txt";
jsonPar.flashShiftVersion = 639;                           % First version of FLASH sequence with built-in off-center correction for P2ROUD
jsonPar.flashExternalNavVersion = 710;                     % First version of flash with external navigator


% AI 

jsonPar.AImodule = false; 


% Parallel cores

jsonPar.maxNumCores = 8;


% Colors

jsonPar.brightred = 'Red';
jsonPar.red = '#D95319';
jsonPar.brightgreen = 'Green';
jsonPar.green = '#6CB557';
jsonPar.yellow = 'Yellow';
jsonPar.orange = '#FDA400';
jsonPar.blue = '#0073BE';
jsonPar.black = 'Black';
jsonPar.lightblue = '#DEE6EB';
jsonPar.lightgrey = '#F0F0F0';
jsonPar.guiBlack = [0.15,0.15,0.15];
jsonPar.guiGrey = [0.94,0.94,0.94];
jsonPar.guiMiddleGrey = [0.9,0.9,0.9];
jsonPar.guiDarkGrey = [0.5,0.5,0.5];
jsonPar.guiWhite = [1.0,1.0,1.0];
jsonPar.guiBlackBlue = [0.02,0.13,0.3];
jsonPar.traceGreen = [0.4660 0.6740 0.1880];
jsonPar.traceOrange = [0.9290 0.6940 0.1250];


% Icons

jsonPar.amsterdamUmcIcon = 'AmsterdamUMC.png';
jsonPar.mrsolutionsIcon = 'MRSolutionsLogo.gif';
jsonPar.retrospectiveIcon = 'Retrospective.png';
jsonPar.diskIcon = 'disk.png';
jsonPar.filterIcon = 'filter.png';
jsonPar.sortIcon = 'sort.gif';
jsonPar.recoIcon = 'reconstruct.gif';
jsonPar.exitIcon = 'exit.png';
jsonPar.playIcon = 'play.gif';
jsonPar.stopIcon = 'stop.gif';
jsonPar.cwIcon = 'cw.png';
jsonPar.ccwIcon = 'ccw.png';
jsonPar.statusOkayIcon = 'statusokay.gif';
jsonPar.statusWarningIcon = 'statuswarning.gif';
jsonPar.statusErrorIcon = 'statuserror.gif';
jsonPar.statusBusyIcon = 'busy.gif';
jsonPar.statusAutoIcon = 'auto.png';
jsonPar.squareIcon = 'square.gif';
jsonPar.contrastIcon = 'contrast.png';
jsonPar.navIcon = 'nav.gif';


% Plotting

jsonPar.navMovieSpeed = 25;
jsonPar.maxNrSpokesPlot = 500;
jsonPar.traceLength = 60;


% Species

jsonPar.mouse.bandwidthHR = 60;
jsonPar.mouse.HeartWidthEditField = 60;
jsonPar.mouse.bandwidthRR = 20;
jsonPar.mouse.RespWidthEditField = 40;
jsonPar.mouse.respPercentage = 30;
jsonPar.mouse.RespWindowEditField = 30;
jsonPar.mouse.MinHeartRateEditField = 300;
jsonPar.mouse.HeartMinEditField = 300;
jsonPar.mouse.MaxHeartRateEditField = 700;
jsonPar.mouse.HeartMaxEditField = 700;
jsonPar.mouse.MinRespRateEditField = 30;
jsonPar.mouse.RespMinEditField = 30;
jsonPar.mouse.MaxRespRateEditField = 120;
jsonPar.mouse.RespMaxEditField = 120;
jsonPar.mouse.FilterOrderEditField = 4;
jsonPar.mouse.RespShiftEditField = 0;

jsonPar.rat.bandwidthHR = 50;
jsonPar.rat.HeartWidthEditField = 50;
jsonPar.rat.bandwidthRR = 20;
jsonPar.rat.RespWidthEditField = 40;
jsonPar.rat.respPercentage = 30;
jsonPar.rat.RespWindowEditField = 30;
jsonPar.rat.MinHeartRateEditField = 200;
jsonPar.rat.HeartMinEditField = 200;
jsonPar.rat.MaxHeartRateEditField = 500;
jsonPar.rat.HeartMaxEditField = 500;
jsonPar.rat.MinRespRateEditField = 30;
jsonPar.rat.RespMinEditField = 30;
jsonPar.rat.MaxRespRateEditField = 120;
jsonPar.rat.RespMaxEditField = 120;
jsonPar.rat.FilterOrderEditField = 4;
jsonPar.rat.RespShiftEditField = 0;

jsonPar.zebrafish.bandwidthHR = 60;
jsonPar.zebrafish.HeartWidthEditField = 60;
jsonPar.zebrafish.bandwidthRR = 20;
jsonPar.zebrafish.RespWidthEditField = 40;
jsonPar.zebrafish.respPercentage = 30;
jsonPar.zebrafish.RespWindowEditField = 30;
jsonPar.zebrafish.MinHeartRateEditField = 60;
jsonPar.zebrafish.HeartMinEditField = 60;
jsonPar.zebrafish.MaxHeartRateEditField = 300;
jsonPar.zebrafish.HeartMaxEditField = 300;
jsonPar.zebrafish.MinRespRateEditField = 10;
jsonPar.zebrafish.RespMinEditField = 10;
jsonPar.zebrafish.MaxRespRateEditField = 90;
jsonPar.zebrafish.RespMaxEditField = 90;
jsonPar.zebrafish.FilterOrderEditField = 4;
jsonPar.zebrafish.RespShiftEditField = 0;


% Filtering

jsonPar.respHarmonics = 4;                          % Number of higher order harmonics for respiratory frequency
jsonPar.splineFactor = 60;                          % data interpolation factor to prevent navigator discretization by TR
jsonPar.respPercentage = 30;                        % percentage of data discarded during respiration
jsonPar.sgFilterOrder = 3;                          % Savitsky-Golay filter order for filtering navigator
jsonPar.sgFilterLength = 9;                         % Savitsky-Golay filter length for filtering navigator


% Reconstruction

jsonPar.AutoGuessRecoCheckBox = 1;

jsonPar.scoutReco.FramesEditField = 1;
jsonPar.scoutReco.DynamicsEditField = 1;
jsonPar.scoutReco.SharingEditField = 0;
jsonPar.scoutReco.WVxyzEditField = 0.001;
jsonPar.scoutReco.TVxyzEditField = 0.005;
jsonPar.scoutReco.LLRxyzEditField = 0.000;
jsonPar.scoutReco.TVcineEditField = 0.000;
jsonPar.scoutReco.TVdynEditField = 0.000;
jsonPar.scoutReco.ESPIRiTCheckBox = [0, 1];
jsonPar.scoutReco.RingRmCheckBox = 0;
jsonPar.scoutReco.NLMFCheckBox = 0;

jsonPar.systolicReco.FramesEditField = 15;
jsonPar.systolicReco.DynamicsEditField = 1;
jsonPar.systolicReco.SharingEditField = 0;
jsonPar.systolicReco.WVxyzEditField = 0.001;
jsonPar.systolicReco.TVxyzEditField = 0.000;
jsonPar.systolicReco.TVcineEditField = [0.010, 0.10];     % [3D, 2D]
jsonPar.systolicReco.LLRxyzEditField = [0, 0.001, 0.01];  % [off, 3D, 2D]
jsonPar.systolicReco.TVdynEditField = 0.000;
jsonPar.systolicReco.ESPIRiTCheckBox = [0, 1];
jsonPar.systolicReco.RingRmCheckBox = 0;
jsonPar.systolicReco.NLMFCheckBox = 0;

jsonPar.diastolicReco.FramesEditField = 32;
jsonPar.diastolicReco.DynamicsEditField = 1;
jsonPar.diastolicReco.SharingEditField = 0;
jsonPar.diastolicReco.WVxyzEditField = 0.001;
jsonPar.diastolicReco.TVxyzEditField = 0.000;
jsonPar.diastolicReco.TVcineEditField = [0.010, 0.10];     % [3D, 2D]
jsonPar.diastolicReco.LLRxyzEditField = [0, 0.001, 0.01];  % [off, 3D, 2D]
jsonPar.diastolicReco.TVdynEditField = 0.000;
jsonPar.diastolicReco.ESPIRiTCheckBox = [0, 1];
jsonPar.diastolicReco.RingRmCheckBox = 0;
jsonPar.diastolicReco.NLMFCheckBox = 0;

jsonPar.respirationReco.FramesEditField = 32;
jsonPar.respirationReco.DynamicsEditField = 1;
jsonPar.respirationReco.SharingEditField = 0;
jsonPar.respirationReco.WVxyzEditField = 0.001;
jsonPar.respirationReco.TVxyzEditField = 0.000;
jsonPar.respirationReco.TVcineEditField = [0.010, 0.10];     % [3D, 2D]
jsonPar.respirationReco.LLRxyzEditField = [0, 0.001, 0.01];  % [off, 3D, 2D]
jsonPar.respirationReco.TVdynEditField = 0.000;
jsonPar.respirationReco.ESPIRiTCheckBox = [0, 1];
jsonPar.respirationReco.RingRmCheckBox = 0;
jsonPar.respirationReco.NLMFCheckBox = 0;

jsonPar.realtimeReco.FramesEditField = 2;
jsonPar.realtimeReco.DynamicsEditField = 1;
jsonPar.realtimeReco.SharingEditField = 3;
jsonPar.realtimeReco.WVxyzEditField = 0.000;
jsonPar.realtimeReco.TVxyzEditField = 0.01;
jsonPar.realtimeReco.TVcineEditField = 0.001; 
jsonPar.realtimeReco.LLRxyzEditField = [0 0.05]; % [no-GPU GPU]
jsonPar.realtimeReco.TVdynEditField = 0.200;
jsonPar.realtimeReco.ESPIRiTCheckBox = [0, 1];
jsonPar.realtimeReco.RingRmCheckBox = 0;
jsonPar.realtimeReco.NLMFCheckBox = 1;

jsonPar.vfaReco.FramesEditField = 6;
jsonPar.vfaReco.DynamicsEditField = 1;
jsonPar.vfaReco.SharingEditField = 0;
jsonPar.vfaReco.WVxyzEditField = 0.002;
jsonPar.vfaReco.TVxyzEditField = 0;
jsonPar.vfaReco.TVcineEditField = 0.01; 
jsonPar.vfaReco.LLRxyzEditField =[0 0.05]; % [no-GPU GPU]
jsonPar.vfaReco.TVdynEditField = 0.001;
jsonPar.vfaReco.ESPIRiTCheckBox = [0, 1];
jsonPar.vfaReco.RingRmCheckBox = 0;
jsonPar.vfaReco.NLMFCheckBox = 1;

jsonPar.multiEchoReco.FramesEditField = 12;
jsonPar.multiEchoReco.DynamicsEditField = 1;
jsonPar.multiEchoReco.SharingEditField = 0;
jsonPar.multiEchoReco.WVxyzEditField = 0.002;
jsonPar.multiEchoReco.TVxyzEditField = 0;
jsonPar.multiEchoReco.TVcineEditField = 0.01; 
jsonPar.multiEchoReco.LLRxyzEditField =[0 0.05]; % [no-GPU GPU]
jsonPar.multiEchoReco.TVdynEditField = 0.000;
jsonPar.multiEchoReco.ESPIRiTCheckBox = [0, 1];
jsonPar.multiEchoReco.RingRmCheckBox = 0;
jsonPar.multiEchoReco.NLMFCheckBox = 0;


% Number of junk samples table [no_samples,gd_no_samples]

jsonPar.gdTable = [25,14; 50,32; 75,32; 100,40; 150,16; 200,20; 400,10; 500,8; 800,5; 1000,4; 2000,2; 4000,1];


% AI parameters

jsonPar.trainNumbers = {'m1','m2','m3','m4','m6'};                  % Folders with DRIM training data [1-1.5][1.5-2.5][2.5-3.5][3.5-4.5][4.5>]
jsonPar.checkpoints = {'50000','50000','50000','50000','50000'};    % Training checkpoint numbers
jsonPar.neededModules = {"tensorboard",...                          % Modules needed for deep learning reconstruction
            "tensorboard-data-server","imageio",...
            "matplotlib","matplotlib-inline",...
            "opencv-python","scikit-image","pillow",...
            "h5py","torch","torchvision","torchmetrics",...
            "tqdm","pandas","scipy","opencv-python",...
            "interval","unzip","python","nibabel","tensorboard"};


% Bart

jsonPar.totalVariation = 'T';                                       % Total variation (T) or total generalized variation (G)


% Image 

jsonPar.maxImageValue = 32767;                                      % Maximum image value
jsonPar.maxKspaceValue = 32767;                                     % Max k-space value


% Files to ignore during import

jsonPar.ignoreFiles = ["cardtrigpoints","resptrigpoints","respwindow","heartrate","resprate",...
    "recoparameters","recoparmeters","retro","p2roud","retroData"];



%% Write to JSON file

jsonText = jsonencode(jsonPar, 'PrettyPrint', true);
fid = fopen('defaultRetroSettings.json', 'w');
fwrite(fid, jsonText, 'char');
fclose(fid);

clearvars;