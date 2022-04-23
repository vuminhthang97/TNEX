FROM alpine
FROM httpd:latest
CMD ["echo", "Hello world!"]

FROM ubuntu
RUN apt-get update
RUN apt-get install curl -y\&& apt-get install nano -y


WORKDIR /var/www/html
# Copy tất cả các file trong thư mục hiện tại (.)  vào WORKDIR
ADD . /var/www/html

#Thiết lập khi tạo container từ image sẽ mở cổng 80
# ở mạng mà container nối vào
EXPOSE 8800

# Khi chạy container tự động chạy ngay httpd
ENTRYPOINT ["/usr/sbin/httpd"]

#chạy terminate
CMD ["-D", "FOREGROUND"]