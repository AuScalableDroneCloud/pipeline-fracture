%--------------------------------------------------------------------------
% This script performs the post-processing on the binarized ridges. The
% operations performed are segmentation using Otsu thresholding,
% skeletonization and polyline fitting
%
% binary ridge image -> segmentation -> skeletonization -> poyline fitting
%
% Calls functions from Geom2D Toolbox by David Legland, CoSHREM Toolbox
% by Rafael Reisenhofer and MATLAB Image Processing Toolbox
%       -> CSHRMgetOverlay.m (from CoSHREM Toolbox)
%       -> polynomialCurveSetFit.m (from Geom2D Toolbox)
%       -> drawPolynomialCurve.m (from Geom2D Toolbox)
%       -> minDistancePoints.m (from Geom2D Toolbox)
%       -> parametrize.m (from Geom2D Toolbox)
%       -> polynomialCurveFit.m (from Geom2D Toolbox)
%       -> polynomialCurvePoint.m (from Geom2D Toolbox)
%       -> polynomialCurveSetFit.m (from Geom2D Toolbox)
%
%         MODIFIED by Geoffry Squire and Dr. Ulrich Kelka September 2021
%--------------------------------------------------------------------------

function Ridge_Post_Processing(BIN_FILES, BIN_PATH, IMG_FILES, IMG_PATH) 
    replicatePath = repmat(cellstr(BIN_PATH),size(BIN_FILES));
    InFileList = strcat(replicatePath,BIN_FILES);          
    funPath = fileparts(which('Ridge_Post_Processing.m'));
    addpath(genpath(funPath)); 
    clear replicatePath;
%% select source images to create overlays for visual comparison        
    replicatePath = repmat(cellstr(IMG_PATH),size(IMG_FILES));
    InFileList2 = strcat(replicatePath,IMG_FILES);    
    funPath2 = fileparts(which('Ridge_Post_Processing.m'));
    addpath(genpath(funPath2)); 
    clear replicatePath;
%%  create directories to save images corresponding to post-processing steps
    
     mkdir(strcat(BIN_PATH,'Segmented_Ridges_Overlay'))
     mkdir(strcat(BIN_PATH,'Fitted_Curves'))

%%  loop that performs the post-processing steps
for m = 1:size(InFileList,2)
    tic   
    % counter
    disp(' ');
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
    disp(['This is image ' num2str(m) ' out of ' num2str(size(InFileList, 2))]);
    disp(['Performing Otsu Thresholding Segmentation for Image  ' num2str(m) ' out of ' num2str(size(InFileList, 2))]);
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
    
    % read the binarized ridge image
    imageIN = imread(InFileList{m});
    
    % read the source image
    imageIN_1 = imread(InFileList2{m});                
    imageIN_1 = rgb2gray(imageIN_1); 
    imageIN_4 = imageIN;
  
    % estimating background using morphological opening
    background = imopen(imageIN,strel('disk',5));
    
    % subtracting the background image from the original image
    imageIN_2 = imageIN - background;
   
    % increasing the image contrast
    imageIN_3 = imadjust(imageIN_2);


    % thresholding the image, set number of pixels to be removed, P = 10
    bw = imbinarize(imageIN_3);
    bw = bwareaopen(bw, 10);
   
    % identifying objects within the image
    cc = bwconncomp(bw, 4);
    
    % creating labels
    labeled = labelmatrix(cc);
    whos labeled;
    RGB_label = label2rgb(labeled, @spring, 'c', 'shuffle');
    
    % removing isolated clusters
    Pixel_List = cc.PixelIdxList';
    for j=1:length(Pixel_List)
     Pixel_List_length(j,1)=length(Pixel_List{j,1});
    end
    imageIN_4 = labeled;

    Isolation_Threshold = 10;
    for j=1:length(Pixel_List)
     if Pixel_List_length(j,1)< Isolation_Threshold
      imageIN_4(imageIN_4==j)=0;
     end
    end
    
    j=1;
    tic
    for k=1:length(Pixel_List)
     [r,c]=find(imageIN_4==k);
     if isempty(r)~=1  
      Cluster_List{j,1}=[r c];
      j=j+1;
     end
    end
    toc
    
   %  converting to image
   imageIN_4(imageIN_4>0)=255;
   % imageIN_4=mat2gray(imageIN_4);
   imageIN_1=double(255 * mat2gray(imageIN_1));
   imageIN_4=double(255 * mat2gray(imageIN_4));
   %imshow(imageIN_1)
 
   % creating overlay of segmented ridges on the source image and saving it
   overlay_plus_segmented_ridges=CSHRMgetOverlay(imageIN_1,imageIN_4);
   imwrite(overlay_plus_segmented_ridges, strcat(BIN_PATH,'Segmented_Ridges_Overlay/',BIN_FILES{m}));

     
   % Skeletonizing the segmented ridges  and
   % calculating branch points and end points for each cluster
   skelImg   = bwmorph(imageIN_4, 'thin', 'inf');
   branchImg = bwmorph(skelImg, 'branchpoints');
   endImg    = bwmorph(skelImg, 'endpoints');

   [row, column] = find(endImg);
   endPts        = [row column];

   [row, column] = find(branchImg);
   branchPts     = [row column];
   cNumBranchPoints = length(branchPts);
   
   %  writing the skeletonized ridges to the output folders
   disp(['Writing the Skeletons for Image ' num2str(m) ' out of ' num2str(size(InFileList, 2))]);  
   skelImg2 = mat2gray(skelImg);
  % imwrite(~skelImg2, strcat(pathname,'Skeletons/',InFileListShort2{m}));
   
   % concatenating the clusters into one large matrix
    Clusters=Cluster_List{1,1};
    for i=2:length(Cluster_List)
     A= Cluster_List{i,1};
     Clusters=[Clusters;A];
    end

   % finding the branch points and end points associated with each cluster
   % and storing them in two additional columns
   % clusters which have no branches and end points are identified and
   % stored in no_branches and no_ends arrays
   
   y=1;
   z=1;
   for i=1:length(Cluster_List)
    Cluster_List{i,2}=intersect(branchPts,Cluster_List{i,1},'rows') ;
    Cluster_List{i,3}=intersect(endPts,Cluster_List{i,1},'rows') ;
     if isempty(Cluster_List{i,2})==1
      no_branches(y,1) = i;
      y=y+1;
     end
     if isempty(Cluster_List{i,3})==1
      no_ends(z,1) = i;
      z=z+1;
     end   
   end

   % removing clusters that have no end points. It seems these clusters
   % are very close to existing clusters and escape the bwmorph function
   if exist('Nodes','var')==1
    length_Cluster_List = length(Cluster_List);
    idx = find(no_ends==length_Cluster_List); 
    Cluster_List(no_ends(1:idx),:)=[];
   end 
   
   % fitting curves through the clusters using function from David Legland.
   % compute coeffs of each individual branch and returns a matrix of
   % labels for each fitted curve. I use this function for the time being.
   % It does not use the endPts and branchPts calculated using the
   % bwmorph call but calculates end points and branch points using bwlabel,
   % bwconncomp and regionprops functions.
   skelImg = imrotate(skelImg,180);
   [coeffs, curve_matrix] = polynomialCurveSetFit(skelImg, 5);
   
     
   %writing the polynomial fit to a table and stored in the output folder
   if isempty(coeffs)~=1
     
    % Obtaining the polynomial points for each fitted curve using coeffs for
     %each curve
    for n = 1:length(coeffs)
     Poly_Points{n,1}= drawPolynomialCurve([0 1], coeffs{n});
    end
     
    % converting cell array to a table
    for i=1:length(Poly_Points)
      Poly_Points_Table_Header {i}= (['Polyline_' ,num2str(i)]);
    end
    Poly_Points_Table=cell2table(Poly_Points);
       
    OutFileName = IMG_FILES{m};
    OutFileName = OutFileName(1:length(OutFileName)-4);
    OutFileName = strcat(BIN_PATH,'Fitted_Curves/',OutFileName);
    disp(['Writing Polyline Points for Image ' num2str(m) ' out of ' num2str(size(InFileList2, 2))]);
    save(OutFileName,'Poly_Points_Table');
    writetable(Poly_Points_Table,OutFileName);
   
   else
      disp(['Image ' num2str(i) ' is empty. No Polylines to write' ]); 
   writetable(Poly_Points_Table,OutFileName);
   end   
   toc
end
end
