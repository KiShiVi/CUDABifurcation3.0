% a = csvread('C:\Users\KiShiVi\Desktop\matZ.csv');
% 
% figure(1);
% plot(a(:,1), a(:,2), 'r.', 'MarkerSize', 0.1);
% 
a = csvread('C:\Users\KiShiVi\Desktop\mat.csv');
% hold on;

%figure(2);
%semilogx(a(:,1),a(:,2), 'r.', 'MarkerSize', 2);
plot(a(:,1), a(:,2), 'r.', 'MarkerSize', 2);
%title('CT = 200; TT = 100; h = 0.005, IC = {0.2, 0.2, 0.2}, Parameterts = {0.5, 10, 28, 2.5} ','interpreter','latex','FontSize', 24);
xlabel('c','interpreter','latex','FontSize', 24);
ylabel('X','interpreter','latex','FontSize', 24);
set(gcf, 'Position', [1 100 1280 640]);
set(gca,'FontSize',20);
set(gca,'TickLabelInterpreter','latex');

%exportgraphics(gcf,'bifuraction.png','Resolution',300);

%exportgraphics(gcf,'bifuraction.pdf','Resolution',300);

%b = csvread('C:\Users\KiShiVi\Desktop\mat1.csv');

%hold on;
%figure(2);
%plot(b(:,1), b(:,2), 'r.', 'MarkerSize', 1);

% x = linspace(0.05, 0.35, 1000)
% image(x, x, a)

% pcolor(a);
% axis ij;
% axis square;
% grid minor