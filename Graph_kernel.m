clear all;
data = load('C:\Users\lzheng43\Dropbox\MRCGAN_network\output_dir_bc\all.mat');
c = 0.95;
BA = data.BA;
GAE = double(data.GAE);
music = double(data.music);
original = double(data.original);
ER = data.ER;
Netgan = data.netgan;

ER = BLin_W2P(ER,-1);
display('ER')
GAE = BLin_W2P(GAE,-1);
display('GAE')
BA = BLin_W2P(BA,-1);
display('BA')
music = BLin_W2P(music,-1);
display('MUSIC')
original = BLin_W2P(original,-1);
display('ORG')
netgan = BLin_W2P(Netgan,-1);
display('Netgan')

GAE = double(GAE + GAE')/2;
BA = (BA + BA')/2;
ER = (ER + ER')/2;
music = double(music + music')/2;
original = double(original + original')/2;
netgan = double(netgan + netgan')/2;

sim1 = GS_RW_Plain(music, original,c,[0 0 0 1]);
display('MUSIC')
sim2 = GS_RW_Plain(ER, original,c,[0 0 0 1]);
display('ER')
sim3 = GS_RW_Plain(BA, original,c,[0 0 0 1]);
display('BA')
sim4 = GS_RW_Plain(GAE, original,c,[0 0 0 1]);
display('GAE')
sim5 = GS_RW_Plain(netgan, original,c,[0 0 0 1]);
display('netgan')

