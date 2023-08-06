css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://4.bp.blogspot.com/-cvjeZMIgcJQ/VzD8PkZvO2I/AAAAAAAA_2k/LaSQWXMFG1A1VMm0wSK0MVppDsOGs_s-wCKgB/s1600/doraemon.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}. Further information can be found at {source}..., page {page}}  
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.pinimg.com/originals/94/45/e6/9445e6aca8e0c09943b971e890a027f7.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''