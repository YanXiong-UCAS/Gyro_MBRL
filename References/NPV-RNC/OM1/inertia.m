function M = inertia(q,p)
s2 = sin(q(2)); c2 = cos(q(2));
s3 = sin(q(3)); c3 = cos(q(3));

a1 = p.Jc - p.Kc;
a2 = p.Jd - p.Id;

Ma = blkdiag(0,0,0,p.Ka);
Mb = blkdiag(0,0,p.Jb,p.Ib*s3^2+p.Kb*c3^2);
Mc = [0,0,0,0;
      0,p.Ic,0,-p.Ic*s3;
      0,0,p.Jc*c2^2+p.Kc*s2^2,a1*s2*c2*c3;
      0,-p.Ic*s3,a1*s2*c2*c3,p.Ic*s3^2+(p.Jc*s2^2+p.Kc*c2^2)*c3^2];
Md = [p.Jd,0,p.Jd*c2,p.Jd*s2*c3;
      0,p.Id,0,-p.Id*s3;
      p.Jd*c2,0,p.Id*s2^2+p.Jd*c2^2,a2*s2*c2*c3;
      p.Jd*s2*c3,-p.Id*s3,a2*s2*c2*c3,p.Id*s3^2+(p.Id*c2^2+p.Jd*s2^2)*c3^2];
  
M = Ma+Mb+Mc+Md;
end