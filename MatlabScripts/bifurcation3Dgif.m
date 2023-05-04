b = csvread('C:\Users\KiShiVi\Desktop\mat.csv');
len = length(b(1,:)) -1;
x = linspace(b(1,1), b(1,2), len);
y = linspace(b(2,1), b(2,2), len);
z = linspace(b(3,1), b(3,2), len);

%x_1 = linspace(b(1,1), b(1,2), len1);
%y_1 = linspace(b(2,1), b(2,2), len1);
%z_1 = linspace(b(3,1), b(3,2), len1);
b(1,:) = [];b(1,:) = [];b(1,:) = [];

arr = zeros(length(b(1,:)) - 1, length(b(1,:)) - 1, length(b(1,:)) - 1);
a_im = zeros(length(b(1,:)) - 1,length(b(1,:)) - 1);

% for i = 1:len
% for j = 1:len
% for k = 1:len
% if (arr(i,j,k) > 20)
% arr(i,j,k) = 20;
% end
% end
% end
% end

for i = 1:len
arr(i,:,:) = b(1:length(b(1,:)) - 1, 1:length(b(1,:)) - 1);
b(1:length(b(1,:)) - 1,:) = [];
end


figure (1)
set(gcf, 'Position', [50, 200, 650, 500])

for i = 1:len
img = reshape(arr(i,:,:),len,len);
figure (1)
imagesc(x,y,img);
TITLE = ['a = ', num2str(z(i))];title(TITLE);
xlabel("h");ylabel("b");
set(gca,'XScale','log');
set(gca,'YDir','normal');
colorbar;
% ylim([1 1200]);
%caxis([1 80]);
%pause(0.5);
%colormap("turbo");
set(gca,'ColorScale','log');
drawnow

end

% for i = 1:len
% img = reshape(arr(:,i,:),len,len);
%
% figure (1)
% imagesc(img);
% imagesc(x,z,img);
% TITLE = ['b = ', num2str(y_1(i))];title(TITLE);
% xlabel("a");ylabel("c");
% set(gca,'YDir','normal');
% colorbar;
% colormap("turbo");
% set(gca,'ColorScale','log');
% drawnow
% % pause(0.5);
% end

% for i = 1:len
% img = reshape(arr(:,:,i),len,len);
%
% figure (1)
% imagesc(img);
% imagesc(y,z,img);
% TITLE = ['a = ', num2str(x_1(i))];title(TITLE);
% xlabel("b");ylabel("c");
% set(gca,'YDir','normal');
% colorbar;
% colormap("turbo");
% set(gca,'ColorScale','log');
% drawnow
% % pause(0.5);
% end