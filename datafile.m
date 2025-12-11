clear;
TX=100;
TY=100;
TZ=100;
PX=6;
PY=8;
PZ=6;

G=zeros(TX,TY,TZ);
r=35;

for i=1:TX
     for j=1:TY
          for k=1:TZ
              if (i-0.5-TX/2.0)^2+(j-0.5-TY/2.0)^2+(k-0.5-TZ/2.0)^2<r^2
                  G(i,j,k)=1;
              end
          end
     end
end

isosurface(G(2:TX-1,2:TY-1,2:TZ-1));
mkdir('data');
for a=1:PX*PY*PZ
        rankz=ceil(a/(PX*PY));
        ranky=ceil((a-(rankz-1)*PX*PY)/PX);
        rankx=a-(rankz-1)*PX*PY-(ranky-1)*PX;
        rank(a,1)=rankx;
        rank(a,2)=ranky;
        rank(a,3)=rankz;

        if rankx<=mod(TX,PX)
            lengthx=ceil(TX/PX);
            startx=(rankx-1)*lengthx;
        else
            lengthx=floor(TX/PX);
            startx=(rankx-1)*lengthx+mod(TX,PX);
        end

        if ranky<=mod(TY,PY)
            lengthy=ceil(TY/PY);
            starty=(ranky-1)*lengthy;
        else
            lengthy=floor(TY/PY);
            starty=(ranky-1)*lengthy+mod(TY,PY);
        end

        if rankz<=mod(TZ,PZ)
            lengthz=ceil(TZ/PZ);
            startz=(rankz-1)*lengthz;
        else
            lengthz=floor(TZ/PZ);
            startz=(rankz-1)*lengthz+mod(TZ,PZ);
        end

        start(a,1)=startx;
        start(a,2)=starty;
        start(a,3)=startz;
        length(a,1)=lengthx;
        length(a,2)=lengthy;
        length(a,3)=lengthz;

        NX=lengthx+2;
        NY=lengthy+2;
        NZ=lengthz+2;

        data=zeros(NX,NY,NZ);

        for i=1:NX
            for j=1:NY
                for k=1:NZ
                    id=startx+i-1;
                    if id==0
                        id=TX;
                    end
                    if id==TX+1
                        id=1;
                    end

                    jd=starty+j-1;
                    if jd==0
                        jd=TY;
                    end
                    if jd==TY+1
                        jd=1;
                    end

                    kd=startz+k-1;
                    if kd==0
                        kd=TZ;
                    end
                    if kd==TZ+1
                        kd=1;
                    end

                    data(i,j,k)=G(id,jd,kd);
                end
            end
        end

        mpirank=num2str(a-1,'%04d');
        filename=['./data/data',mpirank,'.dat'];
        fid=fopen(filename,'w');
        for i=1:NX
            for j=1:NY
                for k=1:NZ
                    fprintf(fid,'%i\n',data(i,j,k));
                end
            end
        end
        fclose(fid);
end