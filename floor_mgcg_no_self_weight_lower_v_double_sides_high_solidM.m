%function top3D125(nelx,nely,nelz,volfrac,penal,rmin,ft,ftBC,eta,beta,move,maxit)
clear;
nl=7;
nelx=20*2^(nl-1);
nely=10*2^(nl-1);
nelz=1*2^(nl-1);  %2.5
% volfrac=0.0;
volfrac0=0.12;
%penal=3;
penal=5;
rmin=4;
ft=3;
ftBC='N';
eta=0.4;
move=0.15;
maxit=40;
beta=2;

% ---------------------------- PRE. 1) MATERIAL AND CONTINUATION PARAMETERS
E0 = 1;                                                                    % Young modulus of solid
Emin = 1e-9;                                                               % Young modulus of "void"
nu = 0.3;                                                                  % Poisson ratio
penalCnt = { 1, 1, 25, 0.25 };                                             % continuation scheme on penal
betaCnt  = { 1, 1, 25,    2 };                                             % continuation scheme on beta
if ftBC == 'N', bcF = 'symmetric'; else, bcF = 0; end                      % filter BC selector
% ----------------------------------------- PRE. 2) DISCRETIZATION FEATURES
nEl = nelx * nely * nelz;                                                  % number of elements          #3D#
nodeNrs = int32( reshape( 1 : ( 1 + nelx ) * ( 1 + nely ) * ( 1 + nelz ), ...
    1 + nely, 1 + nelz, 1 + nelx ) );                                      % nodes numbering             #3D#
cVec = reshape( 3 * nodeNrs( 1 : nely, 1 : nelz, 1 : nelx ) + 1, nEl, 1 ); %                             #3D#
% cMat = cVec+int32( [0,1,2,3*(nely+1)*(nelz+1)+[0,1,2,-3,-2,-1],-3,-2,-1,3*(nely+...
%    1)+[0,1,2],3*(nely+1)*(nelz+2)+[0,1,2,-3,-2,-1],3*(nely+1)+[-3,-2,-1]]);% connectivity matrix         #3D#
nDof = ( 1 + nely ) * ( 1 + nelz ) * ( 1 + nelx ) * 3;                     % total number of DOFs        #3D#
Ke = 1/(1+nu)/(2*nu-1)/144 *( [ -32;-6;-6;8;6;6;10;6;3;-4;-6;-3;-4;-3;-6;10;...
    3;6;8;3;3;4;-3;-3; -32;-6;-6;-4;-3;6;10;3;6;8;6;-3;-4;-6;-3;4;-3;3;8;3;...
    3;10;6;-32;-6;-3;-4;-3;-3;4;-3;-6;-4;6;6;8;6;3;10;3;3;8;3;6;10;-32;6;6;...
    -4;6;3;10;-6;-3;10;-3;-6;-4;3;6;4;3;3;8;-3;-3;-32;-6;-6;8;6;-6;10;3;3;4;...
    -3;3;-4;-6;-3;10;6;-3;8;3;-32;3;-6;-4;3;-3;4;-6;3;10;-6;6;8;-3;6;10;-3;...
    3;8;-32;-6;6;8;6;-6;8;3;-3;4;-3;3;-4;-3;6;10;3;-6;-32;6;-6;-4;3;3;8;-3;...
    3;10;-6;-3;-4;6;-3;4;3;-32;6;3;-4;-3;-3;8;-3;-6;10;-6;-6;8;-6;-3;10;-32;...
    6;-6;4;3;-3;8;-3;3;10;-3;6;-4;3;-6;-32;6;-3;10;-6;-3;8;-3;3;4;3;3;-4;6;...
    -32;3;-6;10;3;-3;8;6;-3;10;6;-6;8;-32;-6;6;8;6;-6;10;6;-3;-4;-6;3;-32;6;...
    -6;-4;3;6;10;-3;6;8;-6;-32;6;3;-4;3;3;4;3;6;-4;-32;6;-6;-4;6;-3;10;-6;3;...
    -32;6;-6;8;-6;-6;10;-3;-32;-3;6;-4;-3;3;4;-32;-6;-6;8;6;6;-32;-6;-6;-4;...
    -3;-32;-6;-3;-4;-32;6;6;-32;-6;-32]+nu*[ 48;0;0;0;-24;-24;-12;0;-12;0;...
    24;0;0;0;24;-12;-12;0;-12;0;0;-12;12;12;48;0;24;0;0;0;-12;-12;-24;0;-24;...
    0;0;24;12;-12;12;0;-12;0;-12;-12;0;48;24;0;0;12;12;-12;0;24;0;-24;-24;0;...
    0;-12;-12;0;0;-12;-12;0;-12;48;0;0;0;-24;0;-12;0;12;-12;12;0;0;0;-24;...
    -12;-12;-12;-12;0;0;48;0;24;0;-24;0;-12;-12;-12;-12;12;0;0;24;12;-12;0;...
    0;-12;0;48;0;24;0;-12;12;-12;0;-12;-12;24;-24;0;12;0;-12;0;0;-12;48;0;0;...
    0;-24;24;-12;0;0;-12;12;-12;0;0;-24;-12;-12;0;48;0;24;0;0;0;-12;0;-12;...
    -12;0;0;0;-24;12;-12;-12;48;-24;0;0;0;0;-12;12;0;-12;24;24;0;0;12;-12;...
    48;0;0;-12;-12;12;-12;0;0;-12;12;0;0;0;24;48;0;12;-12;0;0;-12;0;-12;-12;...
    -12;0;0;-24;48;-12;0;-12;0;0;-12;0;12;-12;-24;24;0;48;0;0;0;-24;24;-12;...
    0;12;0;24;0;48;0;24;0;0;0;-12;12;-24;0;24;48;-24;0;0;-12;-12;-12;0;-24;...
    0;48;0;0;0;-24;0;-12;0;-12;48;0;24;0;24;0;-12;12;48;0;-24;0;12;-12;-12;...
    48;0;0;0;-24;-24;48;0;24;0;0;48;24;0;0;48;0;0;48;0;48 ] );             % elemental stiffness matrix  #3D#
Ke0( tril( ones( 24 ) ) == 1 ) = Ke';
Ke0 = reshape( Ke0, 24, 24 );
Ke0 = Ke0 + Ke0' - diag( diag( Ke0 ) );                                    % recover full matrix
% ----------------------------- PRE. 3) LOADS, SUPPORTS AND PASSIVE DOMAINS
[ pasS, pasV ] = deal( [], [] );                                            % passive solid and void elements
elmNrs = int32( reshape( 1 : nelx*nely*nelz ,nely, nelz, nelx));            % elements numbering             #3D#
floor_thickness_rabar = round(0.046 / 0.4 * nelz);
floor_thickness_cover = round(0.02 / 0.4 * nelz);
step_size_x = round(0.28 / 8.0 * nelx );
step_size_y = round(0.10 / 8.0 * nelx );
pasS1 = double(reshape(elmNrs(1:nely,1:floor_thickness_cover, 1: nelx),[],1));
pasS2 = double(reshape(elmNrs(1:nely,nelz -floor_thickness_cover+1:nelz, 1: nelx),[],1));
pasS3 = [];
x1 = round(0.05 / 8.0 * nelx);
while x1 < nelx
    if x1+step_size_x > nelx
        break;
    end
    pasS3 = union(pasS3, double(reshape(elmNrs(1:nely,1: floor_thickness_rabar...
       , x1+1: x1+floor_thickness_rabar),[],1)));
    x1 = x1 + step_size_x;
end

y2 = round(0.05 / 4.0 * nely);
while y2 <= nely
    if y2+step_size_y > nely
        break;
    end
    pasS3 = union(pasS3, double(reshape(elmNrs(y2+1 : y2+floor_thickness_rabar,1 : floor_thickness_rabar ...
       , 1:nelx),[],1)));
    y2 = y2 + step_size_y;
end
x_3 = round(nelx / 3);
solid_z = round(0.08 / 0.4 * nelz);
pasS4 = double(reshape(elmNrs(1: nely , nelz - solid_z + 1 : nelz,x_3 + 1: x_3 *2),[],1));
pasS = union(pasS4,union(pasS3,union(pasS1, pasS2)));


fixed11 = 3*nodeNrs( 1:nely+1, 1, [1,nelx + 1])-2;
fixed12=3*nodeNrs( 1:nely+1, 1, [1,nelx + 1])-1;
fixed13=3*nodeNrs( 1:nely+1, 1, [1,nelx + 1]);
fixed = double(union(fixed13,union(fixed11,fixed12)));
% 
% [y_indices, z_indices, x_indices] = ind2sub([nely+1, nelz+1, nelx+1],fixed) ;
% 
% % Display the 3D coordinates of each element in pasV1
% pasV1_index = [x_indices, y_indices, z_indices];
% 
% % Use the scatter3 function to draw the corresponding units
% scatter3(pasV1_index(:, 1), pasV1_index(:, 2), pasV1_index(:, 3), 'filled');
% xlabel('i');
% ylabel('j');
% zlabel('k');
% title('Visualization of pasV1\_index');
% grid on;

free = setdiff( 1 : nDof, fixed );                                         % set of free DOFs
act = setdiff( ( 1 : nEl )', union( pasS, pasV ) );                        % set of active d.v.
% --------------------------------------- PRE. 4) DEFINE IMPLICIT FUNCTIONS
prj = @(v,eta,beta) (tanh(beta*eta)+tanh(beta*(v(:)-eta)))./...
    (tanh(beta*eta)+tanh(beta*(1-eta)));                                   % projection
deta = @(v,eta,beta) - beta * csch( beta ) .* sech( beta * ( v( : ) - eta ) ).^2 .* ...
    sinh( v( : ) * beta ) .* sinh( ( 1 - v( : ) ) * beta );                % projection eta-derivative 
dprj = @(v,eta,beta) beta*(1-tanh(beta*(v-eta)).^2)./(tanh(beta*eta)+tanh(beta*(1-eta)));% proj. x-derivative
cnt = @(v,vCnt,l) v+(l>=vCnt{1}).*(v<vCnt{2}).*(mod(l,vCnt{3})==0).*vCnt{4};
% -------------------------------------------------- PRE. 5) PREPARE FILTER
[dy,dz,dx]=meshgrid(-ceil(rmin)+1:ceil(rmin)-1,...
    -ceil(rmin)+1:ceil(rmin)-1,-ceil(rmin)+1:ceil(rmin)-1 );
h = max( 0, rmin - sqrt( dx.^2 + dy.^2 + dz.^2 ) );                        % conv. kernel                #3D#
Hs = imfilter( ones( nely, nelz, nelx ), h, bcF );                         % matrix of weights (filter)  #3D#
dHs = Hs;
% ------------------------ PRE. 6) ALLOCATE AND INITIALIZE OTHER PARAMETERS
[ x, dsK, dV ] = deal( zeros( nEl, 1 ) );                                  % initialize vectors
                                             % derivative of volume
%x( act ) = (volfrac*( nEl - length(pasV) ) - length(pasS) )/length( act );% volume fraction on active set
%x( act ) = (volfrac*( nEl - numel(pasV) ) - numel(pasS) )/numel( act );    % volume fraction on active set

% x( act ) = volfrac0/10;
% x (iniAct) = volfrac0*10;
x( act ) = volfrac0;
x( pasS ) = 1.0;                                                            % set x = 0.05 on pasS set
x( pasV ) = 1e-6;
F=zeros(nDof,1);
% F0 = zeros(nDof,1);
direction3 = 3;
g3 = 10.0;
lcDof1 = 3 * nodeNrs(1:nely+1,nelz+1,1:nelx+1);
F(lcDof1) = 40.0;
% Assembly_F_1(F,pasS, cVec,g3, nelx, nely, nelz,direction3);
F(fixed)=0.0;
%Assembly_F0_ Before loop
% volfrac0 =(nEl * 0.1010 -  numel(pasS)*1.0+numel(pasV)*1e-6) / numel(act);
volfrac=(numel(act)*volfrac0+numel(pasS)*1.0+numel(pasV)*1e-6)/nEl;
dV( act, 1 ) = 1/nEl/volfrac; 
[ xPhys, xOld, ch, loop, U ] = deal( x, 1, 1, 0, zeros( nDof, 1 ) );       % old x, x change, it. counter, U

% ================================================= START OPTIMIZATION LOOP
nu=0.3; nswp=16; printLev=1; hx=1; hy=1; hz=1; cgtol=1e-5; cgmax=400;
while ch > 1e-8 && loop < maxit
  tic
  loop = loop + 1;                                                         % update iter. counter
  % ----------- RL. 1) COMPUTE PHYSICAL DENSITY FIELD (AND ETA IF PROJECT.)
  xTilde = imfilter( reshape( x, nely, nelz, nelx ), h, bcF ) ./ Hs;       % filtered field              #3D#
  xPhys( act ) = xTilde( act );                                            % reshape to column vector
  if ft > 1                                                                % compute optimal eta* with Newton
      f = ( mean( prj( xPhys, eta, beta ) ) - volfrac )  * (ft == 3);      % function (volume)
      while abs( f ) > 1e-6           % Newton process for finding opt. eta
          eta = eta - f / mean( deta( xPhys, eta, beta ) );
          f = mean( prj( xPhys, eta, beta ) ) - volfrac;
      end
      dHs = Hs ./ reshape( dprj( xPhys, eta, beta ), nely, nelz, nelx );   % sensitivity modification    #3D#
      xPhys = prj( xPhys, eta, beta );                                     % projected (physical) field
  end
  ch = norm( xPhys - xOld ) ./ nEl;
  xOld = xPhys;

  % -------------------------- RL. 2) SETUP AND SOLVE EQUILIBRIUM EQUATIONS
  %-sK = ( Emin + xPhys.^penal * ( E0 - Emin ) );
  dsK( act ) = -penal * ( E0 - Emin ) * xPhys( act ) .^ ( penal - 1 );

  CX=Emin+xPhys.^penal*(E0-Emin);
  nfd=length(fixed);
  [cgres,cgiters]=mgcg9(Ke0,F,U,CX,fixed',nfd,nelx,nely,nelz,nl,cgtol,cgmax,nswp, printLev, nu, hx, hy, hz);
  if cgiters<0, break; end
  fprintf('MGCG relres: %4.2e iters: %4i \n',cgres,cgiters);
  
  % ------------------------------------------ RL. 3) COMPUTE SENSITIVITIES
  % dc = dsK .* sum( ( U( cMat ) * Ke0 ) .* U( cMat ), 2 );                  % derivative of compliance
  dc = sum_UKU7(Ke0,dsK, U, cVec, nelx, nely, nelz);
  dc = imfilter( reshape( dc, nely, nelz, nelx ) ./ dHs, h, bcF );         % filter objective sens.      #3D#
  dV0 = imfilter( reshape( dV, nely, nelz, nelx ) ./ dHs, h, bcF );        % filter compliance sens.     #3D#
  % ----------------- RL. 4) UPDATE DESIGN VARIABLES AND APPLY CONTINUATION
  xT = x( act );
  [ xU, xL ] = deal( xT + move, xT - move );                               % current upper and lower bound
  ocP = xT .* sqrt( - dc( act ) ./ dV0( act ) );                           % constant part in resizing rule
  l = [ 0, mean( ocP ) / volfrac ];                                        % initial estimate for LM
  while ( l( 2 ) - l( 1 ) ) / ( l( 2 ) + l( 1 ) ) > 1e-4                   % OC resizing rule
      lmid = 0.5 * ( l( 1 ) + l( 2 ) );
      x( act ) = max( max( min( min( ocP / lmid, xU ), 1 ), xL ), 0 );
      if mean( x ) > volfrac, l( 1 ) = lmid; else, l( 2 ) = lmid; end
  end
  [penal,beta] = deal(cnt(penal,penalCnt,loop), cnt(beta,betaCnt,loop));   % apply conitnuation on parameters
  % -------------------------- RL. 5) PRINT CURRENT RESULTS AND PLOT DESIGN
  fprintf( 'It.:%5i C:%6.5e V:%7.3f ch.:%0.2e penal:%7.2f beta:%7.1f eta:%7.2f\n', ...
      loop, F'*U, mean(xPhys(:)), ch, penal, beta, eta);
  toc
end
%Specify the file name to save (file path can be customized)
filename = strcat('xphys_one_floor_lower_high_solidM_',num2str(volfrac0),'_',num2str(loop),'.mat');
% Use the save function to save variables to a file
save(filename, 'xPhys','-v7.3');
filename1 = strcat('one_floor_U_lower_high_solidM_',num2str(volfrac0),'_',num2str(loop),'.mat');
% Use the save function to save variables to a file
save(filename1, 'U','-v7.3');
isovals = shiftdim( reshape( xPhys, nely, nelz, nelx ), 2 );
isovals = smooth3( isovals, 'box', 1 );
patch(isosurface(isovals, .5),'FaceColor',[192, 192, 192]/255,'EdgeColor','none'); %'b'
patch(isocaps(isovals, .5),'FaceColor','r','EdgeColor','none');
drawnow; view( [ 145, 25 ] ); axis equal tight off; camlight;
savefig(strcat('QNC_V1_one_floor_lower_high_solidM_',num2str(loop),'.fig'));
%end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Matlab code was written by F. Ferrari, O. Sigmund                   %
% Dept. of Solid Mechanics-Technical University of Denmark,2800 Lyngby (DK)%
% Please send your comments to: feferr@mek.dtu.dk                          %
%                                                                          %
% The code is intended for educational purposes and theoretical details    %
% are discussed in the paper Ferrari, F. Sigmund, O. - A new generation 99 %
% line Matlab code for compliance Topology Optimization and its extension  %
% to 3D, SMO, 2020                                                         %
%                                                                          %
% The code as well as a postscript version of the paper can be             %
% downloaded from the web-site: http://www.topopt.dtu.dk                   %
%                                                                          %
% Disclaimer:                                                              %
% The authors reserves all rights but do not guaranty that the code is     %
% free from errors. Furthermore, we shall not be liable in any event       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
