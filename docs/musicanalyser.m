clear all
close all

%% Import Audio
filepath = "RAM/";
[Y, Fs]=audioread(filepath + "01 Daft Punk - Give Life Back to Music.flac");
%Sum stereo to Mono
Y = (Y(:,1)+Y(:,2))/2;

disp('Import: Complete')
%% Octave band filtering
%starts octbank with the unfiltered signal in column one
OB = Y; %OB stands for Octave Bank which will store all the filtered time series data
%Adds filtered signal to octbank such that column two is 31.250
for Fc = [31.250 62.500 125 250 500 1000 2000 4000 8000 16000];
    N = 1; %octave filter, 3 for 3rd octave
    [B,A] = octdsgn(Fc,Fs,N);
    oct = filter(B,A,Y);
    OB = [OB oct];
    disp('Processing Octave: ' + Fc)
end

disp('Octave band Filtering: Complete')
%% Octave band
freqlims = [10 Fs/2];   
bandsperoct = 3;        %3 = 3rd Octave, uses a higher order octave filter to show data
weighting = "none";     %A, C, none
opts = {'FrequencyLimits',freqlims,'BandsPerOctave',bandsperoct,'Weighting',weighting};
[p,cf] = poctave(OB,Fs,opts{:});
pdB = 10*log10(p/1.);

%Plot data
semilogx(cf, pdB)
axis([20 20000 -60 0])
grid on
legend('Full Spectrum','31.250', '62.500', '125 250', '500', '1000', '2000', '4000', '8000', '16000','Location','bestoutside')

disp('Octave Band Graph: Complete')
%% Analysis - Max & RMS
%use ob for octavebank
for n = 1:11
    %Calculate the Maximum Signal Peak
    OBmax(n) = max(abs(OB(:,n)));
    OBmaxdB(n) = 20*log10(OBmax(n));
    %calculate the line at which the signal is above for 10% of the time
    %??
    %Calcualte the RMS
    OBrms(n) = rms(OB(:,n));
    OBrmsdB(n) = 20*log10(OBrms(n));
    %Calculate the dynamic range
    OBdr(n) = OBrms(n)/OBmax(n);
    OBdrdB(n) = 20*log10(OBdr(n));
end

disp('Analysis Complete')

%% Calculate Octave band distributions
h1 = histogram(OB(:,1));
h2 = histogram(OB(:,2));
h3 = histogram(OB(:,3));
h4 = histogram(OB(:,4));
h5 = histogram(OB(:,5));
h6 = histogram(OB(:,6));
h7 = histogram(OB(:,7));
h8 = histogram(OB(:,8));
h9 = histogram(OB(:,9));
h10 = histogram(OB(:,10));
h11 = histogram(OB(:,11));

%% Graphs
Fc = [0 31.250 62.500 125 250 500 1000 2000 4000 8000 16000];
semilogx(Fc, OBmaxdB, Fc, OBrmsdB); 
grid on
axis([20 20000 -40 0])

