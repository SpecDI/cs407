$(document).ready(function() {

  $('.file-upload').change(function() {
      $('#form-upload').submit();
      $('#file-label').fadeOut("slow", "linear");
  });


  $('.upload').hide();
  $('.users').hide();
  $('.tracking').hide();
  $('.action').hide();

  $('.icon-explain').hide()

  speed = 1500

      $('.upload').fadeIn(speed, function(){
        $('.users').fadeIn(speed, function(){
          $('.tracking').fadeIn(speed, function(){
            $('.action').fadeIn(speed, function(){

            });
          });
        });
      });

        //  alert("hi");



    $('#upload_icon').hover(function(){
      $('.icon-text').toggle();
      $('.icon-explain span').text("Upload Your Drone Video To Be Processed");
      $('.icon-explain').toggle();

    });


    $('#user_icon').hover(function(){
      $('.icon-text').toggle();
      $('.icon-explain span').text("Uses YOLO To Detect Each Person In The Frame");
      $('.icon-explain').toggle();
    });


    $('#track_icon').hover(function(){
      $('.icon-text').toggle();
      $('.icon-explain span').text("Uses DeepSort To Link Each Person Between Frames");
      $('.icon-explain').toggle();
    });

    $('#action_icon').hover(function(){
      $('.icon-text').toggle();
      $('.icon-explain span').text("Uses An LSTM To Detect Actions");
      $('.icon-explain').toggle();
    });


});
