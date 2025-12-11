clear;

rng('shuffle');

cube_size = 50;
num_points = 6;
min_distance = 20;

coordinates = zeros(num_points, 3);

coordinates(1, :) = cube_size * rand(1, 3);

for i = 2:num_points
    while true
        new_point = cube_size * rand(1, 3);
        distances = sqrt(sum((coordinates(1:i-1, :) - new_point).^2, 2));
        if all(distances > min_distance)
            coordinates(i, :) = new_point; 
            break;
        end
    end
end

for i=1:num_points
    coordinates(i,:)=coordinates(i,:)+25;
end


filename='./particle.dat';
        fid=fopen(filename,'w');
        for i=1:num_points
            fprintf(fid,'%f %f %f\n',coordinates(i,1), coordinates(i,2), coordinates(i,3));
        end
        fclose(fid);

