% a = csvread('C:\Users\KiShiVi\Desktop\mat.csv');
% 
% x = linspace(a(1,1), a(1,2), 10);
% y = linspace(a(2,1), a(2,2), 10);
% a(1,:) = []
% a(1,:) = [];
% 
% figure(1);
% imagesc(x, y, a);
% colorbar


a = csvread('C:\Users\KiShiVi\Desktop\mat.csv');
x = linspace(a(1,1),a(1,2),10);
y = linspace(a(2,1),a(2,2),10)
a(1,:) = [];
a(1,:) = [];
a(:,end) = [];
figure
% imagesc(x,y,a);%medfilt2(ref_CompCD,[m n]);
imagesc(x,y,medfilt2(a,[1 1]));
%set(gca,'YDir','normal');
%xlabel("a");
%ylabel("b");

xlabel('V','interpreter','latex','FontSize', 24);
ylabel('d','interpreter','latex','FontSize', 24);
set(gcf, 'Position', [1 100 1280 640]);
set(gca,'FontSize',20);
set(gca,'TickLabelInterpreter','latex');



colormap("hot")
colorbar;

%exportgraphics(gcf,'bifuraction.png','Resolution',300);

%exportgraphics(gcf,'bifuraction.pdf','Resolution',300);

%set(gca,'ColorScale','log');
%caxis([1 110]);



%caxis([200 1000]);
%colormap('jet');


%caxis([0, 1]);


%b = csvread('C:\Users\KiShiVi\Desktop\mat1.csv');

%x1 = linspace(b(1,1), b(1,2), 10);
%y1 = linspace(b(2,1), b(2,2), 10);
%b(1,:) = [];
%b(1,:) = [];

%figure(2);
%imagesc(x1, y1, b);
%colorbar;
%caxis([0, 1000]);


% pcolor(a);
% axis ij;
%axis square;
% grid minor