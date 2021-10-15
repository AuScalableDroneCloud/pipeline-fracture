%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SHEARLET SYSTEM and DETECTION PARAMETERS (RIDGES)
%  waveletEffSupp
%  gaussianEffSup
%  scalesPerOctave
%  shearLevel
%  alpha
%  octaves
%
% DETECTION PARAMETERS
% detection type
% minContrast
% offset
%
% POLYLINE TO SHP
% simplification (Douglas-Peuker)
% topological correction parameters (angle, distance, vertex degree)
% projection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function main(folder, row, col)%shearlet system ensemble, ridge detection parameters, intesity threshhols (Otsu threshold)  
    [IMG_FILES, IMG_PATH] = Ridge_Ensemble_Generator(folder, row,col);
    [BIN_FILES, BIN_PATH] = Ridge_Ensemble_Reader(folder);
    Ridge_Post_Processing(BIN_FILES, BIN_PATH, IMG_FILES, IMG_PATH)
end