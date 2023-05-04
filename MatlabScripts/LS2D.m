a = csvread('C:\Users\KiShiVi\Desktop\mat.csv1.csv');
b = csvread('C:\Users\KiShiVi\Desktop\mat.csv2.csv');
c = csvread('C:\Users\KiShiVi\Desktop\mat.csv3.csv');
x = linspace(a(1,1),a(1,2),10);
y = linspace(a(2,1),a(2,2),10)
a(1,:) = [];
a(1,:) = [];
a(:,end) = [];
b(1,:) = [];
b(1,:) = [];
b(:,end) = [];
c(1,:) = [];
c(1,:) = [];
c(:,end) = [];

%d = a + b + c;
d = 2 + (a + b) ./ abs(c);


figure(1);
imagesc(x,y,medfilt2(a,[1 1]));
title('a');
%xlabel('V','interpreter','latex','FontSize', 24);
%ylabel('d','interpreter','latex','FontSize', 24);
set(gcf, 'Position', [1 100 1280 640]);
set(gca,'FontSize',20);
set(gca,'TickLabelInterpreter','latex');
colormap("turbo")
colorbar;

figure(2);
imagesc(x,y,medfilt2(b,[1 1]));
title('b');
%xlabel('V','interpreter','latex','FontSize', 24);
%ylabel('d','interpreter','latex','FontSize', 24);
set(gcf, 'Position', [1 100 1280 640]);
set(gca,'FontSize',20);
set(gca,'TickLabelInterpreter','latex');
colormap("turbo")
colorbar;

figure(3);
imagesc(x,y,medfilt2(c,[1 1]));
title('c');
%xlabel('V','interpreter','latex','FontSize', 24);
%ylabel('d','interpreter','latex','FontSize', 24);
set(gcf, 'Position', [1 100 1280 640]);
set(gca,'FontSize',20);
set(gca,'TickLabelInterpreter','latex');
colormap("turbo")
colorbar;

figure(4);
imagesc(x,y,medfilt2(d,[1 1]));
title('d');
%xlabel('V','interpreter','latex','FontSize', 24);
%ylabel('d','interpreter','latex','FontSize', 24);
set(gcf, 'Position', [1 100 1280 640]);
set(gca,'FontSize',20);
set(gca,'TickLabelInterpreter','latex');
colormap("turbo")
colorbar;


