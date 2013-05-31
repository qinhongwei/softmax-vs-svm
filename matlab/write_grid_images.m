%{
Copyright (C) 2013 Yichuan Tang. contact: tang at cs.toronto.edu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
%}

%{
Data  should be n x d, where n is the number of images, d is the serialized
image data, row major. imagedim is nRows xnCols of image size
griddim: images in big image
borderwidth: number of pixels between images
borderval: [0 to 1] value of border between images
This function assumes that the images are row by row serialization of Data(i,:) into a 2D image

RGB is supported!!
%}
function [image] = write_grid_images(Data, imagedim, griddim, borderwidth, borderval)


image = borderval*ones(griddim(1)*imagedim(1)+borderwidth*(griddim(1)+1), ...
                griddim(2)*imagedim(2)+borderwidth*(griddim(2)+1),size(Data,3));
            
for i = 1:griddim(1)    
    for j = 1:griddim(2)
                
        ival = (i-1)*(imagedim(1)+borderwidth)+1+borderwidth;
        jval = (j-1)*(imagedim(2)+borderwidth)+1+borderwidth;
               
        im_2d = reshape(Data( (i-1)*griddim(2)+j, :,:), imagedim(2) , imagedim(1) ,size(Data,3));        
        %the imagedim(2) first hen imagedim(1) is because we do a
        %transpose at the end
        
        if length(size(im_2d)) == 3
            im_2d = permute(im_2d, [2 1 3]);
        else
           im_2d = im_2d'; 
        end
        
        image(ival:ival+imagedim(1)-1, jval:jval+imagedim(2)-1,:) = im_2d;      
    end
end


