function jsonPars = readSettingsFile(versionTag)

% Define destination folder and file name
settingsFolder = fullfile(tempdir, 'Retrospective', versionTag);
settingsFile = fullfile(settingsFolder, 'retroSettings.json');

% Check if the settings file exists, if not make a new one
if ~isfile(settingsFile)

    % Make sure the folder exists
    if ~exist(settingsFolder, 'dir')
        mkdir(settingsFolder);
    end

    % The default settings file
    defaultFile = which(fullfile('defaultRetroSettings.json'));
 
    % Copy the default to the tempdir versioned location
    copyfile(defaultFile, settingsFile);

end

% Read the settings file
fid = fopen(settingsFile, 'r');
jsonRawText = fread(fid, inf, 'char=>char')';
fclose(fid);
jsonPars = jsondecode(jsonRawText);

end
