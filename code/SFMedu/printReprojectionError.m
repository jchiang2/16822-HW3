function proj_error = printReprojectionError(graph)
	% Convert from R, t pairs to angle-axis representation
    nCam=length(graph.frames);
    Mot = zeros(3,2,nCam);
    for camera=1:nCam
        Mot(:,1,camera) = RotationMatrix2AngleAxis(graph.Mot(:,1:3,camera));
        Mot(:,2,camera) = graph.Mot(:,4,camera);
    end

    Str = graph.Str;
    f  = graph.f;

    % assume px, py=0
    px = 0;
    py = 0;
    residuals = reprojectionResidual(graph.ObsIdx,graph.ObsVal,px,py,f,Mot,Str);

	proj_error = sqrt(sum(residuals.^2) / length(residuals));
	fprintf('The reprojection error is %f\n', proj_error);
end