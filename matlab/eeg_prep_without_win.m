% LOADING SIGNALS

for i = 101:108
    filename = sprintf('%dFS2.mat', i); 
    data = load(filename); 
    fieldnames_data = fieldnames(data); 
    content = data.(fieldnames_data{1}); 

    % Assign to dynamically named variable
    assignin('base', sprintf('sig%d', i - 100), content);
end

%NO 109, 110

for i = 111:128
    filename = sprintf('%dFS2.mat', i); 
    data = load(filename); 
    fieldnames_data = fieldnames(data);
    content = data.(fieldnames_data{1}); 
    assignin('base', sprintf('sig%d', i - 102), content);
end
 
%NO RECORDINGS OF P 129, 130

for i = 131:180
    filename = sprintf('%dFS2.mat', i);
    data = load(filename); 
    fieldnames_data = fieldnames(data); 
    content = data.(fieldnames_data{1});
    assignin('base', sprintf('sig%d', i - 104), content); 
end

%NO RECORDINGS OF P 182

%WRONG TIMING IN 184

%NO RECORDINGS OF P 186

%NO RECORDINGS OF P 188

for i = 189:200
    filename = sprintf('%dFS2.mat', i); 
    data = load(filename); 
    fieldnames_data = fieldnames(data); 
    content = data.(fieldnames_data{1}); 
    assignin('base', sprintf('sig%d', i - 112), content); 
end

%---------------------------------
% NORMALISE SIGS TO THE SHORTEST SIG LENGTH

min_rows = inf;

for i = 1:88
    array = eval(sprintf('sig%d', i));
    [rows, cols] = size(array); 
    min_rows = min(min_rows, size(array(:,1))); 
end

disp(['Smallest number of rows: ', num2str(min_rows)]);

%sig78 WAS TERMINATED BEFORE 4 MINUTES RESULTING IN HAVING 105K POINTS
%INSTEAD OF 120K

%DECIDED TO ELIMINATE SIGNALS 183 AND 184

sigs = cell(1, 88);
for i = 1:88
    var_name = sprintf('sig%d', i);
    if evalin('base', sprintf('exist(''%s'', ''var'')', var_name))
        sigs{i} = evalin('base', var_name);
    else
        warning('Variable %s does not exist.', var_name);
    end
end

min_rows = min(cellfun(@(x) size(x, 1), sigs));
disp(['Smallest number of rows 222: ', num2str(min_rows)]);

for i = 1:88
    sigs{i} = sigs{i}(1:min_rows, :); % Change each array to 119520 length
end

%---------------------
%ARTIFACT DETECTION

fs = 500; % Sampling frequency (Hz)

% Artifact detection parameters
amplitude_threshold = 30000; 
gradient_threshold = 75;


clean_signals = cell(1, 88);
num_ar = zeros(88, 17); 
num_ar_diff = zeros(88, 17);

for i = 1:88
    for j = 1:8
        signal = sigs{i}(:, j);
    
        % 1. Amplitude-based artifact detection
        high_amplitude_idx = find(abs(signal) > amplitude_threshold); 
        num_artifacts = length(high_amplitude_idx);
        num_ar(i, j) = num_artifacts;
    
        % 2. Gradient-based artifact detection
        gradient_signal = diff(signal); 
        high_gradient_idx = find(abs(gradient_signal) > gradient_threshold); 
        num_ar(i, j + 9) = length(high_gradient_idx);
 
        % Combine artifact indices
        artifact_idx = unique([high_amplitude_idx; high_gradient_idx]);

        % Optional: Remove or mark artifacts
        clean_signal(:, j) = signal;
        for idx = artifact_idx'
            % Check for out-of-bounds indices
            if idx > 1 && idx < length(signal)
                % Replace with the average of the neighbors
                clean_signal(idx, j) = mean([clean_signal(idx - 1, j), clean_signal(idx + 1, j)]);
            elseif idx == 1
                % For the first element
                clean_signal(idx, j) = clean_signal(idx + 1, j);
            elseif idx == length(signal)
                % For the last element
                clean_signal(idx, j) = clean_signal(idx - 1, j);
            end
        end
    end
    clean_signals{i} = clean_signal; % Store the cleaned signal
end

disp('Artifact detection complete!');

%------------------------
%BAND SEPARATION

bands = {
    'Delta', [0.5 4];
    'Theta', [4 8];
    'Alpha', [8 13];
    'Beta', [13 30];
    'Gamma', [30 100];
};

t = 1:min_rows
filtered_signals = cell(88*5, 9);
r = 0

for k = 1:88 
    for j = 1:8
        for i = 1:size(bands, 1)
            band_name = bands{i, 1};
            freq_range = bands{i, 2};
            
    
            % Design a bandpass Butterworth filter
            [b, a] = butter(4, freq_range / (fs / 2), 'bandpass'); % Normalize by Nyquist frequency
    
            filtered_signals{i + r, 1} = bands{i, 1};
            filtered_signals{i + r, j+1} = filtfilt(b, a, clean_signals{k}(:, j));
        end
    end
    r = r + 5;
end

%------------------------
% CORRELATION COMPUTATION

sig_corr = cell(44*5, 9);
p = 0;

for k = 1:220
    c = k + p*5;
    for j = 2:9
        disp(c);
        sig_corr{k, j} = corr(filtered_signals{c, j}, filtered_signals{c + 5, j});
    end
    if mod(c, 5) == 0
        p = p + 1;
    end
end

for k = 1:220
    sig_corr{k, 1} = filtered_signals{k, 1};
end

    
writecell(sig_corr, 'correlations_array.csv');

disp('Cell array written to correlations_array.csv');
