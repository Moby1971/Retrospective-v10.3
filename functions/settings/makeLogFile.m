function logFileName = makeLogFile(versionTag)

% Define destination folder and file name
logFolder = fullfile(tempdir, 'Retrospective', versionTag);

% Make sure the folder exists
if ~exist(logFolder, 'dir')
    mkdir(logFolder);
end

% Date and time tag
logTag = string(datetime);
logTag = strrep(logTag," ","-");
logTag = strrep(logTag,":","-");

% Log file folder / name
logFileName = fullfile(logFolder, strcat("retrospective-log-",logTag,".txt"));

end
